import {
	type FetchFunction,
	withoutTrailingSlash,
} from "@ai-sdk/provider-utils"
import {
	type AuthLoaderOptions,
	type EffectiveAuth,
	loadAuthTokens,
} from "./auth.js"
import { collectCompletedResponseFromSse } from "./sse.js"
import { CodexResponsesState } from "./state.js"

export const DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
const DEFAULT_CODEX_INSTRUCTIONS = ""

export type CodexOAuthSettings = Omit<AuthLoaderOptions, "fetch"> & {
	baseURL?: string
	codexVersion?: string
	fetch?: FetchFunction
	headers?: Record<string, string>
	instructions?: string
	store?: boolean
	responsesState?: CodexResponsesState | false
}

type RequestParts = {
	url: string
	method?: string
	headers: Headers
	body?: BodyInit | null
	signal?: AbortSignal | null
}

class AuthManager {
	private current?: EffectiveAuth
	private inflight?: Promise<EffectiveAuth>
	private readonly settings: CodexOAuthSettings
	private readonly fetch: FetchFunction

	constructor(settings: CodexOAuthSettings, fetch: FetchFunction) {
		this.settings = settings
		this.fetch = fetch
	}

	async ensure(): Promise<EffectiveAuth> {
		if (this.inflight) {
			return this.inflight
		}

		this.inflight = loadAuthTokens({
			authFilePath: this.settings.authFilePath,
			clientId: this.settings.clientId,
			issuer: this.settings.issuer,
			tokenUrl: this.settings.tokenUrl,
			fetch: this.fetch,
			ensureFresh: this.settings.ensureFresh,
			now: this.settings.now,
		})
			.then((auth) => {
				this.current = auth
				this.inflight = undefined
				return auth
			})
			.catch((error) => {
				this.inflight = undefined
				throw error
			})

		return this.inflight
	}

	async headers(): Promise<Record<string, string>> {
		const auth = this.current ?? (await this.ensure())
		return {
			Authorization: `Bearer ${auth.accessToken}`,
			"chatgpt-account-id": auth.accountId,
			"OpenAI-Beta": "responses=experimental",
		}
	}
}

const pickFetch = (customFetch?: FetchFunction): FetchFunction => {
	if (typeof customFetch === "function") {
		return customFetch
	}

	if (typeof globalThis.fetch === "function") {
		return globalThis.fetch
	}

	throw new Error("A fetch implementation is required for Codex OAuth.")
}

export const getDefaultCodexInstructions = (): string => {
	return DEFAULT_CODEX_INSTRUCTIONS
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
	typeof value === "object" && value !== null && !Array.isArray(value)

const toolCallTypes = new Set([
	"function_call",
	"custom_tool_call",
	"local_shell_call",
	"shell_call",
	"apply_patch_call",
])

const toolOutputTypes = new Set([
	"function_call_output",
	"custom_tool_call_output",
	"local_shell_call_output",
	"shell_call_output",
	"apply_patch_call_output",
])

const MAX_TOOL_OUTPUT_LENGTH = 10_485_760
const TOOL_OUTPUT_TRUNCATION_BUFFER = 1024

const truncateLongString = (value: string, maxLength: number): string => {
	if (value.length <= maxLength) {
		return value
	}
	const keep = maxLength - TOOL_OUTPUT_TRUNCATION_BUFFER
	const head = Math.floor(keep / 2)
	const tail = keep - head
	const removed = value.length - keep
	return `${value.slice(0, head)}\n\n[... truncated ${removed} characters to fit the 10MB tool output limit ...]\n\n${value.slice(value.length - tail)}`
}

const sanitizeOversizedToolOutputs = (body: Record<string, unknown>): void => {
	if (!Array.isArray(body.input)) {
		return
	}

	for (const item of body.input) {
		if (!isRecord(item) || typeof item.type !== "string") {
			continue
		}
		if (!toolOutputTypes.has(item.type)) {
			continue
		}

		const output = item.output
		if (
			typeof output === "string" &&
			output.length > MAX_TOOL_OUTPUT_LENGTH
		) {
			item.output = truncateLongString(output, MAX_TOOL_OUTPUT_LENGTH)
			continue
		}

		if (Array.isArray(output)) {
			for (const part of output) {
				if (!isRecord(part)) {
					continue
				}
				if (
					typeof part.text === "string" &&
					part.text.length > MAX_TOOL_OUTPUT_LENGTH
				) {
					part.text = truncateLongString(part.text, MAX_TOOL_OUTPUT_LENGTH)
				}
			}
		}
	}
}

const validateStatelessResponsesInput = (
	body: Record<string, unknown>,
): string | undefined => {
	if (body.store === true || !Array.isArray(body.input)) {
		return undefined
	}

	const seenToolCallIds = new Set<string>()
	for (const [index, item] of body.input.entries()) {
		if (!isRecord(item)) {
			continue
		}

		if (item.type === "item_reference" && typeof item.id === "string") {
			return `Unresolved item_reference ${item.id} at input[${index}] cannot be sent when store is false. Replay the full item content or enable store.`
		}

		if (typeof item.type !== "string" || typeof item.call_id !== "string") {
			continue
		}

		if (toolCallTypes.has(item.type)) {
			seenToolCallIds.add(item.call_id)
			continue
		}

		if (toolOutputTypes.has(item.type) && !seenToolCallIds.has(item.call_id)) {
			return `No preceding tool call found for ${item.type} with call_id ${item.call_id} at input[${index}]. Stateless requests must keep each assistant tool call before its tool result; reset this conversation or replay the missing tool call.`
		}
	}

	return undefined
}

const invalidResponsesRequest = (message: string): Response =>
	new Response(
		JSON.stringify({
			error: {
				message,
				type: "invalid_request_error",
				param: "input",
				code: null,
			},
		}),
		{
			status: 400,
			headers: {
				"Content-Type": "application/json",
			},
		},
	)

const resolveBaseURL = (baseURL?: string): string => {
	return withoutTrailingSlash(baseURL) ?? DEFAULT_CODEX_BASE_URL
}

const resolveTargetUrl = (input: string, baseURL: string): string => {
	const base = new URL(baseURL)
	const parsed = /^https?:\/\//.test(input)
		? new URL(input)
		: new URL(input, "https://codex.invalid")
	let pathname = parsed.pathname
	const basePath = withoutTrailingSlash(base.pathname) ?? ""

	if (pathname === basePath) {
		pathname = "/"
	} else if (basePath.length > 0 && pathname.startsWith(`${basePath}/`)) {
		pathname = pathname.slice(basePath.length)
	}

	if (pathname === "/v1") {
		pathname = "/"
	} else if (pathname.startsWith("/v1/")) {
		pathname = pathname.slice(3)
	}

	return `${base.origin}${basePath}${pathname}${parsed.search}`
}

const readRequestParts = async (
	input: Parameters<FetchFunction>[0],
	init: Parameters<FetchFunction>[1],
): Promise<RequestParts> => {
	if (input instanceof Request) {
		const headers = new Headers(input.headers)
		if (init?.headers) {
			new Headers(init.headers).forEach((value, key) => {
				headers.set(key, value)
			})
		}

		return {
			url: input.url,
			method: init?.method ?? input.method,
			headers,
			body:
				init?.body ??
				(input.body == null ? undefined : await input.clone().text()),
			signal: init?.signal ?? input.signal,
		}
	}

	return {
		url: String(input),
		method: init?.method,
		headers: new Headers(init?.headers),
		body: init?.body,
		signal: init?.signal,
	}
}

const decodeBody = async (
	body: BodyInit | null | undefined,
): Promise<string | undefined> => {
	if (body == null) {
		return undefined
	}

	if (typeof body === "string") {
		return body
	}

	if (body instanceof URLSearchParams || body instanceof FormData) {
		return undefined
	}

	if (body instanceof ReadableStream) {
		return undefined
	}

	if (body instanceof Blob) {
		return body.text()
	}

	if (body instanceof ArrayBuffer) {
		return new TextDecoder().decode(body)
	}

	if (ArrayBuffer.isView(body)) {
		return new TextDecoder().decode(body)
	}

	return undefined
}

export type NormalizeCodexResponsesBodyOptions = {
	instructions?: string
	forceStream?: boolean
	store?: boolean
}

export const normalizeCodexResponsesBody = (
	body: Record<string, unknown>,
	options: NormalizeCodexResponsesBodyOptions = {},
): Record<string, unknown> => {
	const normalized: Record<string, unknown> = { ...body }
	const instructions =
		typeof normalized.instructions === "string"
			? normalized.instructions
			: (options.instructions ?? getDefaultCodexInstructions())

	normalized.instructions = instructions

	if (normalized.store === undefined) {
		normalized.store = options.store ?? false
	}

	if (options.forceStream) {
		normalized.stream = true
	}

	delete normalized.max_output_tokens

	return normalized
}

type PreparedResponsesRequestBody = {
	body: BodyInit | null | undefined
	requestBody?: Record<string, unknown>
	response?: Response
}

const prepareResponsesRequestBody = async (
	pathname: string,
	headers: Headers,
	body: BodyInit | null | undefined,
	settings: CodexOAuthSettings,
	state: CodexResponsesState | undefined,
): Promise<PreparedResponsesRequestBody> => {
	if (!pathname.endsWith("/responses")) {
		return { body }
	}

	const contentType = headers.get("content-type")
	if (contentType && !contentType.includes("application/json")) {
		return { body }
	}

	const bodyText = await decodeBody(body)
	if (typeof bodyText !== "string") {
		return { body }
	}

	try {
		const parsed = JSON.parse(bodyText)
		if (!isRecord(parsed)) {
			return { body }
		}

		const normalized = normalizeCodexResponsesBody(parsed, {
			instructions: settings.instructions,
			store: settings.store,
		})

		if (state?.requiresCachedState(normalized)) {
			await state.waitForPendingCaptures()
		}

		const expanded = state?.expandRequestBody(normalized) ?? normalized
		sanitizeOversizedToolOutputs(expanded)
		const validationError = validateStatelessResponsesInput(expanded)
		if (validationError != null) {
			return {
				body: null,
				response: invalidResponsesRequest(validationError),
			}
		}

		return {
			body: JSON.stringify(expanded),
			requestBody: expanded,
		}
	} catch {
		return { body }
	}
}

const captureResponsesState = (
	response: Response,
	requestBody: Record<string, unknown> | undefined,
	state: CodexResponsesState | undefined,
): Response => {
	if (
		state == null ||
		requestBody == null ||
		!response.ok ||
		response.body == null
	) {
		return response
	}

	const [returnedBody, cachedBody] = response.body.tee()
	const capturePromise = collectCompletedResponseFromSse(cachedBody)
		.then((completedResponse) => {
			state.rememberResponse(completedResponse, requestBody)
		})
		.catch(() => undefined)
	state.trackPendingCapture(capturePromise)

	return new Response(returnedBody, {
		status: response.status,
		statusText: response.statusText,
		headers: new Headers(response.headers),
	})
}

export const createCodexOAuthFetch = (
	settings: CodexOAuthSettings = {},
): FetchFunction => {
	const fetch = pickFetch(settings.fetch)
	const authManager = new AuthManager(settings, fetch)
	const baseURL = resolveBaseURL(settings.baseURL)
	const responsesState =
		settings.responsesState === false
			? undefined
			: (settings.responsesState ?? new CodexResponsesState())
	const codexFetch: FetchFunction = async (
		input: Parameters<FetchFunction>[0],
		init: Parameters<FetchFunction>[1],
	) => {
		const request = await readRequestParts(input, init)
		const targetUrl = resolveTargetUrl(request.url, baseURL)
		const target = new URL(targetUrl)

		const headers = new Headers(settings.headers)
		request.headers.forEach((value, key) => {
			headers.set(key, value)
		})
		headers.delete("authorization")
		headers.delete("chatgpt-account-id")
		headers.delete("openai-beta")

		const authHeaders = await authManager.headers()
		for (const [key, value] of Object.entries(authHeaders)) {
			headers.set(key, value)
		}

		const preparedBody = await prepareResponsesRequestBody(
			target.pathname,
			headers,
			request.body,
			settings,
			responsesState,
		)

		if (preparedBody.response != null) {
			return preparedBody.response
		}

		const response = await fetch(target.toString(), {
			method: request.method ?? init?.method,
			headers,
			body: preparedBody.body,
			signal: request.signal ?? undefined,
		})

		return captureResponsesState(
			response,
			preparedBody.requestBody,
			responsesState,
		)
	}

	if (typeof fetch.preconnect === "function") {
		codexFetch.preconnect = fetch.preconnect.bind(fetch)
	}

	return codexFetch
}

export type CodexOAuthClient = {
	baseURL: string
	fetch: FetchFunction
	request: (path: string, init?: RequestInit) => Promise<Response>
}

export const createCodexOAuthClient = (
	settings: CodexOAuthSettings = {},
): CodexOAuthClient => {
	const baseURL = resolveBaseURL(settings.baseURL)
	const fetch = createCodexOAuthFetch(settings)

	return {
		baseURL,
		fetch,
		request: (path, init) => fetch(resolveTargetUrl(path, baseURL), init),
	}
}
