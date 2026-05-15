import type {
	OpenAIOAuthServerLogEvent,
	OpenAIOAuthServerLogPayload,
	OpenAIOAuthServerOptions,
} from "./types.js"

export const createRequestLogger = (
	settings: OpenAIOAuthServerOptions,
): ((event: OpenAIOAuthServerLogEvent) => void) | undefined => {
	if (typeof settings.requestLogger === "function") {
		return settings.requestLogger
	}

	if (process.env.CODEX_OPENAI_SERVER_LOG_REQUESTS !== "1") {
		return undefined
	}

	return (event) => {
		console.log(
			JSON.stringify({
				source: "openai-oauth",
				...event,
			}),
		)
	}
}

export const emitRequestLog = (
	logger: ((event: OpenAIOAuthServerLogEvent) => void) | undefined,
	event: OpenAIOAuthServerLogPayload,
) => {
	try {
		logger?.({
			...event,
			timestamp: new Date().toISOString(),
		})
	} catch {}
}
