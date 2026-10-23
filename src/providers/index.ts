import * as http from "node:http";
import * as https from "node:https";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenerativeAI, type Content, type Part } from "@google/generative-ai";
import { CostCalculator, type CostCalculation } from "../cost/index.js";
import { ModelCatalog, type ModelInfo } from "../catalog/index.js";
import type { AppConfig, GeminiConfig, ModelPricing, ProviderType } from "../config/index.js";
import type { GoogleOAuthManager } from "../auth/index.js";
import type { OpenAiToolSchema } from "../tools/index.js";

/**
 * Programmatic interface allowing projects using `@nexustools/aitooling` to define and register
 * their own custom AI service providers via code (`AiProviderRegistry.registerProvider(...)` or
 * `ProviderFactory.registerCustomProvider(...)`).
 *
 * Registered custom providers automatically work across:
 * - `ProviderFactory.createProvider()`
 * - `ModelCatalog.fetchAvailableModels()` & `ModelCatalog.getFallbackModels()`
 * - `MultiProviderToolStreamClient` / `LmStudioAiClient` (with full tool-calling & 3 reasoning levels)
 * - `AiLiveChatController` & multi-provider UI settings dropdowns
 */
export interface CustomAiServiceAdapter {
    readonly providerId: string;
    readonly displayName: string;
    readonly defaultModel?: string;
    readonly defaultBaseUrl?: string;
    readonly requiresApiKey?: boolean;
    readonly supportsBaseUrl?: boolean;
    readonly fallbackModels?: ModelInfo[];
    /**
     * Optional factory for `ProviderFactory.createProvider(config, oauthManager)`.
     */
    createProvider?: (
        config: AppConfig & Record<string, any>,
        oauthManager?: GoogleOAuthManager
    ) => BaseAIProvider;
    /**
     * Optional model discovery callback for `ModelCatalog` and UI model dropdowns.
     */
    fetchModels?: (
        apiKey?: string,
        baseUrl?: string,
        config?: Record<string, any>
    ) => Promise<Array<ModelInfo | string>>;
    /**
     * Optional streaming tool-calling completion handler emitting OpenAI-compatible SSE delta chunks
     * (`{ choices: [{ delta: { content?, reasoning_content?, tool_calls? } }] }`, `{ done: true }`, `{ error?: string }`).
     */
    streamToolChatCompletion?: (
        payload: AiStreamRequestPayload,
        config: Record<string, any>,
        onChunk: (chunk: any) => void,
        signal?: AbortSignal
    ) => Promise<void>;
}

/**
 * Global registry for custom programmatic AI service providers.
 */
export class AiProviderRegistry {
    private static readonly customAdapters: Map<string, CustomAiServiceAdapter> = new Map();

    public static registerProvider(adapter: CustomAiServiceAdapter): void {
        if (!adapter || !adapter.providerId) {
            throw new Error("CustomAiServiceAdapter requires a non-empty providerId.");
        }
        this.customAdapters.set(adapter.providerId, adapter);
    }

    public static unregisterProvider(providerId: string): boolean {
        return this.customAdapters.delete(providerId);
    }

    public static getProvider(providerId: string): CustomAiServiceAdapter | undefined {
        return this.customAdapters.get(providerId);
    }

    public static getAllProviders(): CustomAiServiceAdapter[] {
        return Array.from(this.customAdapters.values());
    }

    public static clearCustomProviders(): void {
        this.customAdapters.clear();
    }
}

export interface ChatImageAttachment {
    mimeType: string;
    base64Data: string;
    label?: string;
}

export interface ChatHistoryItem {
    type: "system" | "user" | "model" | "assistant";
    text: string;
    images?: ChatImageAttachment[];
}

export interface ProviderResponse {
    text: string;
    inputTokens: number;
    outputTokens: number;
    costUSD: number;
}

export type TokenCallback = (token: string) => void;

export abstract class BaseAIProvider {
    abstract readonly providerId: string;
    abstract readonly model: string;

    protected accumulatedInputTokens = 0;
    protected accumulatedOutputTokens = 0;
    protected accumulatedCostUSD = 0;
    protected customPricing?: Record<string, ModelPricing>;

    constructor(customPricing?: Record<string, ModelPricing>) {
        this.customPricing = customPricing;
    }

    abstract prompt(history: ChatHistoryItem[], userPrompt: string): Promise<ProviderResponse>;
    abstract promptStream(history: ChatHistoryItem[], userPrompt: string, onToken: TokenCallback): Promise<ProviderResponse>;

    protected recordUsage(inputTokens: number, outputTokens: number): CostCalculation {
        const costCalc = CostCalculator.calculateCost(this.model, inputTokens, outputTokens, this.customPricing);
        this.accumulatedInputTokens += inputTokens;
        this.accumulatedOutputTokens += outputTokens;
        this.accumulatedCostUSD += costCalc.totalCostUSD;
        return costCalc;
    }

    getCostSummary(): { accumulatedInputTokens: number; accumulatedOutputTokens: number; accumulatedCostUSD: number } {
        return {
            accumulatedInputTokens: this.accumulatedInputTokens,
            accumulatedOutputTokens: this.accumulatedOutputTokens,
            accumulatedCostUSD: this.accumulatedCostUSD
        };
    }

    getCostPromptContext(): string {
        return CostCalculator.formatCostPromptContext(
            this.model,
            { input: this.accumulatedInputTokens, output: this.accumulatedOutputTokens },
            this.accumulatedCostUSD,
            this.customPricing
        );
    }
}

function formatOpenAIMessageContent(item: ChatHistoryItem): any {
    if (!item.images || item.images.length === 0) {
        return item.text;
    }
    const parts: any[] = [];
    if (item.text) {
        parts.push({ type: "text", text: item.text });
    }
    for (const img of item.images) {
        parts.push({
            type: "image_url",
            image_url: {
                url: `data:${img.mimeType};base64,${img.base64Data}`
            }
        });
    }
    return parts;
}

export class OpenAIProvider extends BaseAIProvider {
    readonly providerId: string = "openai";
    readonly model: string;
    protected client: OpenAI;

    constructor(apiKey: string, model = "gpt-4o", customPricing?: Record<string, ModelPricing>, baseUrl?: string) {
        super(customPricing);
        this.model = model;
        this.client = new OpenAI({
            apiKey: apiKey || "dummy-key-for-local",
            baseURL: baseUrl,
            dangerouslyAllowBrowser: true
        });
    }

    async prompt(history: ChatHistoryItem[], userPrompt: string): Promise<ProviderResponse> {
        const [actualModel, thinkingEffort] = this.model.split(":");
        const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

        for (const item of history) {
            const role = item.type === "system" ? "system" : item.type === "model" || item.type === "assistant" ? "assistant" : "user";
            messages.push({ role, content: formatOpenAIMessageContent(item) } as any);
        }

        messages.push({ role: "user", content: userPrompt });

        const payload: any = {
            model: actualModel,
            messages
        };

        if (thinkingEffort && thinkingEffort !== "none") {
            payload.reasoning = { effort: thinkingEffort };
            payload.reasoning_effort = thinkingEffort;
        }

        const response = await this.client.chat.completions.create(payload);

        const text = response.choices[0]?.message?.content || "";
        const inputTokens = response.usage?.prompt_tokens || 0;
        const outputTokens = response.usage?.completion_tokens || 0;
        const usage = this.recordUsage(inputTokens, outputTokens);

        return {
            text,
            inputTokens,
            outputTokens,
            costUSD: usage.totalCostUSD
        };
    }

    async promptStream(history: ChatHistoryItem[], userPrompt: string, onToken: TokenCallback): Promise<ProviderResponse> {
        const [actualModel, thinkingEffort] = this.model.split(":");
        const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

        for (const item of history) {
            const role = item.type === "system" ? "system" : item.type === "model" || item.type === "assistant" ? "assistant" : "user";
            messages.push({ role, content: formatOpenAIMessageContent(item) } as any);
        }

        messages.push({ role: "user", content: userPrompt });

        const payload: any = {
            model: actualModel,
            messages,
            stream: true,
            stream_options: { include_usage: true }
        };

        if (thinkingEffort && thinkingEffort !== "none") {
            payload.reasoning_effort = thinkingEffort;
        }

        const stream = await this.client.chat.completions.create(payload) as unknown as AsyncIterable<OpenAI.Chat.ChatCompletionChunk>;

        let fullText = "";
        let inputTokens = 0;
        let outputTokens = 0;

        for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content || "";
            if (content) {
                fullText += content;
                onToken(content);
            }
            if ((chunk as any).usage) {
                inputTokens = (chunk as any).usage.prompt_tokens || 0;
                outputTokens = (chunk as any).usage.completion_tokens || 0;
            }
        }

        const usage = this.recordUsage(inputTokens, outputTokens);

        return {
            text: fullText,
            inputTokens,
            outputTokens,
            costUSD: usage.totalCostUSD
        };
    }
}

export class GrokProvider extends OpenAIProvider {
    override readonly providerId = "grok";

    constructor(apiKey: string, model = "grok-2", customPricing?: Record<string, ModelPricing>) {
        super(apiKey, model, customPricing, "https://api.x.ai/v1");
    }
}

function formatClaudeMessageContent(item: ChatHistoryItem): any {
    if (!item.images || item.images.length === 0) {
        return item.text;
    }
    const contentBlocks: any[] = [];
    for (const img of item.images) {
        contentBlocks.push({
            type: "image",
            source: {
                type: "base64",
                media_type: img.mimeType,
                data: img.base64Data
            }
        });
    }
    if (item.text) {
        contentBlocks.push({ type: "text", text: item.text });
    }
    return contentBlocks;
}

export class ClaudeProvider extends BaseAIProvider {
    readonly providerId: string = "claude";
    readonly model: string;
    protected client: Anthropic;

    constructor(apiKey: string, model = "claude-3-5-sonnet-20241022", customPricing?: Record<string, ModelPricing>, baseUrl?: string) {
        super(customPricing);
        this.model = model;
        this.client = new Anthropic({
            apiKey: apiKey || "dummy-key",
            baseURL: baseUrl,
            dangerouslyAllowBrowser: true
        });
    }

    async prompt(history: ChatHistoryItem[], userPrompt: string): Promise<ProviderResponse> {
        const [actualModel, thinkingMode] = this.model.split(":");
        let systemPrompt = "";
        const messages: Anthropic.MessageParam[] = [];

        for (const item of history) {
            if (item.type === "system") {
                systemPrompt += (systemPrompt ? "\n\n" : "") + item.text;
            } else {
                const role = item.type === "model" || item.type === "assistant" ? "assistant" : "user";
                messages.push({ role, content: formatClaudeMessageContent(item) });
            }
        }

        messages.push({ role: "user", content: userPrompt });

        const payload: any = {
            model: actualModel,
            max_tokens: 8192,
            system: systemPrompt || undefined,
            messages
        };

        if (thinkingMode === "adaptive") {
            payload.thinking = { type: "adaptive" };
        } else if (thinkingMode && thinkingMode !== "none") {
            payload.thinking = { type: "enabled", budget_tokens: 2048 };
        }

        const response = await this.client.messages.create(payload);

        const text = response.content.map(c => c.type === "text" ? c.text : "").join("\n");
        const inputTokens = response.usage?.input_tokens || 0;
        const outputTokens = response.usage?.output_tokens || 0;
        const usage = this.recordUsage(inputTokens, outputTokens);

        return {
            text,
            inputTokens,
            outputTokens,
            costUSD: usage.totalCostUSD
        };
    }

    async promptStream(history: ChatHistoryItem[], userPrompt: string, onToken: TokenCallback): Promise<ProviderResponse> {
        const [actualModel, thinkingMode] = this.model.split(":");
        let systemPrompt = "";
        const messages: Anthropic.MessageParam[] = [];

        for (const item of history) {
            if (item.type === "system") {
                systemPrompt += (systemPrompt ? "\n\n" : "") + item.text;
            } else {
                const role = item.type === "model" || item.type === "assistant" ? "assistant" : "user";
                messages.push({ role, content: formatClaudeMessageContent(item) });
            }
        }

        messages.push({ role: "user", content: userPrompt });

        const payload: any = {
            model: actualModel,
            max_tokens: 8192,
            system: systemPrompt || undefined,
            messages,
            stream: true
        };

        if (thinkingMode === "adaptive") {
            payload.thinking = { type: "adaptive" };
        } else if (thinkingMode && thinkingMode !== "none") {
            payload.thinking = { type: "enabled", budget_tokens: 2048 };
        }

        const stream = await this.client.messages.stream(payload);

        let fullText = "";
        let inputTokens = 0;
        let outputTokens = 0;

        for await (const event of stream) {
            if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
                fullText += event.delta.text;
                onToken(event.delta.text);
            } else if (event.type === "message_start") {
                inputTokens = event.message.usage?.input_tokens || 0;
            } else if (event.type === "message_delta") {
                outputTokens = event.usage?.output_tokens || 0;
            }
        }

        const usage = this.recordUsage(inputTokens, outputTokens);

        return {
            text: fullText,
            inputTokens,
            outputTokens,
            costUSD: usage.totalCostUSD
        };
    }
}

export class CustomClaudeProvider extends ClaudeProvider {
    override readonly providerId = "custom-claude";

    constructor(baseUrl: string, apiKey: string, model: string, customPricing?: Record<string, ModelPricing>) {
        super(apiKey || "custom-key", model || "custom-claude", customPricing, baseUrl || "http://localhost:8080");
    }
}

const googleOneQuotaResetTimes = new Map<string, number>();

function buildGeminiParts(item: ChatHistoryItem): any[] {
    const parts: any[] = [];
    if (item.text || (!item.images || item.images.length === 0)) {
        parts.push({ text: item.text });
    }
    if (item.images && item.images.length > 0) {
        for (const img of item.images) {
            parts.push({
                inlineData: {
                    mimeType: img.mimeType,
                    data: img.base64Data
                }
            });
        }
    }
    return parts;
}

export class GeminiProvider extends BaseAIProvider {
    readonly providerId: string;
    readonly model: string;
    private readonly config: GeminiConfig;
    private readonly oauthManager?: GoogleOAuthManager;
    private genAI?: GoogleGenerativeAI;
    private isGoogleOnePreferred: boolean;

    constructor(
        config: GeminiConfig,
        oauthManager?: GoogleOAuthManager,
        customPricing?: Record<string, ModelPricing>,
        isGoogleOne = false
    ) {
        super(customPricing);
        this.config = config;
        this.providerId = isGoogleOne ? "gemini-googleone" : "gemini";
        this.model = config.model || "gemini-2.5-flash";
        this.oauthManager = oauthManager;
        this.isGoogleOnePreferred = isGoogleOne;

        if (config.apiKey) {
            this.genAI = new GoogleGenerativeAI(config.apiKey);
        }
    }

    async prompt(history: ChatHistoryItem[], userPrompt: string): Promise<ProviderResponse> {
        const [actualModel, thinkingLevel] = this.model.split(":");
        let systemInstruction = "";
        const contents: { role: string; parts: any[] }[] = [];

        for (const item of history) {
            if (item.type === "system") {
                systemInstruction += (systemInstruction ? "\n\n" : "") + item.text;
            } else {
                const role = item.type === "model" || item.type === "assistant" ? "model" : "user";
                contents.push({ role, parts: buildGeminiParts(item) });
            }
        }

        contents.push({ role: "user", parts: [{ text: userPrompt }] });

        const resetTime = googleOneQuotaResetTimes.get(actualModel) || 0;
        const isQuotaExhausted = Date.now() < resetTime;

        if (this.oauthManager && this.config.oauthTokens && (this.isGoogleOnePreferred || !isQuotaExhausted)) {
            try {
                const accessToken = await this.oauthManager.getValidAccessToken(this.config.oauthTokens);
                return await this.promptViaOAuth(systemInstruction, contents, accessToken, actualModel, thinkingLevel);
            } catch (oauthErr: any) {
                if (oauthErr.message?.includes("429") || oauthErr.message?.includes("quota") || oauthErr.message?.includes("RESOURCE_EXHAUSTED")) {
                    googleOneQuotaResetTimes.set(actualModel, Date.now() + 3600 * 1000);
                }
                if (!this.config.apiKey) throw oauthErr;
            }
        }

        if (!this.genAI && this.config.apiKey) {
            this.genAI = new GoogleGenerativeAI(this.config.apiKey);
        }

        if (!this.genAI) {
            throw new Error("Gemini Provider is not authenticated. Please provide an API key or sign in with Google.");
        }

        const modelParams: any = {
            model: actualModel,
            systemInstruction: systemInstruction || undefined
        };

        if (thinkingLevel && thinkingLevel !== "none") {
            modelParams.generationConfig = {
                thinkingConfig: { thinkingLevel }
            };
        }

        const model = this.genAI.getGenerativeModel(modelParams);
        const chatContents: Content[] = contents.map(c => ({
            role: c.role,
            parts: c.parts as Part[]
        }));

        const result = await model.generateContent({ contents: chatContents });
        const responseText = result.response.text();
        const usageMetadata = result.response.usageMetadata;
        const inputTokens = usageMetadata?.promptTokenCount || 0;
        const outputTokens = usageMetadata?.candidatesTokenCount || 0;
        const usage = this.recordUsage(inputTokens, outputTokens);

        return {
            text: responseText,
            inputTokens,
            outputTokens,
            costUSD: usage.totalCostUSD
        };
    }

    private async promptViaOAuth(
        systemInstruction: string,
        contents: { role: string; parts: any[] }[],
        accessToken: string,
        actualModel: string,
        thinkingLevel?: string
    ): Promise<ProviderResponse> {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(actualModel)}:generateContent`;
        const payload: any = { contents };

        if (systemInstruction) {
            payload.systemInstruction = { parts: [{ text: systemInstruction }] };
        }

        if (thinkingLevel && thinkingLevel !== "none") {
            payload.generationConfig = {
                thinkingConfig: { thinkingLevel }
            };
        }

        const res = await fetch(url, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${accessToken}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`Gemini OAuth API Error (${res.status}): ${errorText}`);
        }

        const data = await res.json();
        const responseText = data.candidates?.[0]?.content?.parts?.map((p: any) => p.text).join("\n") || "";
        const inputTokens = data.usageMetadata?.promptTokenCount || 0;
        const outputTokens = data.usageMetadata?.candidatesTokenCount || 0;
        const usage = this.recordUsage(inputTokens, outputTokens);

        return {
            text: responseText,
            inputTokens,
            outputTokens,
            costUSD: usage.totalCostUSD
        };
    }

    async promptStream(history: ChatHistoryItem[], userPrompt: string, onToken: TokenCallback): Promise<ProviderResponse> {
        const [actualModel, thinkingLevel] = this.model.split(":");
        let systemInstruction = "";
        const contents: { role: string; parts: any[] }[] = [];

        for (const item of history) {
            if (item.type === "system") {
                systemInstruction += (systemInstruction ? "\n\n" : "") + item.text;
            } else {
                const role = item.type === "model" || item.type === "assistant" ? "model" : "user";
                contents.push({ role, parts: buildGeminiParts(item) });
            }
        }

        contents.push({ role: "user", parts: [{ text: userPrompt }] });

        const resetTime = googleOneQuotaResetTimes.get(actualModel) || 0;
        const isQuotaExhausted = Date.now() < resetTime;

        if (this.oauthManager && this.config.oauthTokens && (this.isGoogleOnePreferred || !isQuotaExhausted)) {
            try {
                const accessToken = await this.oauthManager.getValidAccessToken(this.config.oauthTokens);
                return await this.promptStreamViaOAuth(systemInstruction, contents, accessToken, actualModel, onToken, thinkingLevel);
            } catch (oauthErr: any) {
                if (oauthErr.message?.includes("429") || oauthErr.message?.includes("quota") || oauthErr.message?.includes("RESOURCE_EXHAUSTED")) {
                    googleOneQuotaResetTimes.set(actualModel, Date.now() + 3600 * 1000);
                }
                if (!this.config.apiKey) throw oauthErr;
            }
        }

        if (!this.genAI && this.config.apiKey) {
            this.genAI = new GoogleGenerativeAI(this.config.apiKey);
        }

        if (!this.genAI) {
            throw new Error("Gemini Provider is not authenticated. Please provide an API key or sign in with Google.");
        }

        const modelParams: any = {
            model: actualModel,
            systemInstruction: systemInstruction || undefined
        };

        if (thinkingLevel && thinkingLevel !== "none") {
            modelParams.generationConfig = {
                thinkingConfig: { thinkingLevel }
            };
        }

        const model = this.genAI.getGenerativeModel(modelParams);
        const chatContents: Content[] = contents.map(c => ({
            role: c.role,
            parts: c.parts as Part[]
        }));

        const result = await model.generateContentStream({ contents: chatContents });

        let fullText = "";
        let inputTokens = 0;
        let outputTokens = 0;

        for await (const chunk of result.stream) {
            const text = chunk.text();
            if (text) {
                fullText += text;
                onToken(text);
            }
            if (chunk.usageMetadata) {
                inputTokens = chunk.usageMetadata.promptTokenCount || inputTokens;
                outputTokens = chunk.usageMetadata.candidatesTokenCount || outputTokens;
            }
        }

        const usage = this.recordUsage(inputTokens, outputTokens);

        return {
            text: fullText,
            inputTokens,
            outputTokens,
            costUSD: usage.totalCostUSD
        };
    }

    private async promptStreamViaOAuth(
        systemInstruction: string,
        contents: { role: string; parts: any[] }[],
        accessToken: string,
        actualModel: string,
        onToken: TokenCallback,
        thinkingLevel?: string
    ): Promise<ProviderResponse> {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(actualModel)}:streamGenerateContent?alt=sse`;
        const payload: any = { contents };

        if (systemInstruction) {
            payload.systemInstruction = { parts: [{ text: systemInstruction }] };
        }

        if (thinkingLevel && thinkingLevel !== "none") {
            payload.generationConfig = {
                thinkingConfig: { thinkingLevel }
            };
        }

        const res = await fetch(url, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${accessToken}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`Gemini OAuth API Error (${res.status}): ${errorText}`);
        }

        if (!res.body) {
            throw new Error("No response body received from Gemini stream API.");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullText = "";
        let inputTokens = 0;
        let outputTokens = 0;
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const dataStr = line.substring(6).trim();
                    if (dataStr === "[DONE]") continue;

                    try {
                        const data = JSON.parse(dataStr);
                        const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
                        if (text) {
                            fullText += text;
                            onToken(text);
                        }

                        if (data.usageMetadata) {
                            inputTokens = data.usageMetadata.promptTokenCount || inputTokens;
                            outputTokens = data.usageMetadata.candidatesTokenCount || outputTokens;
                        }
                    } catch {
                        // Ignore parse errors for incomplete chunks in SSE
                    }
                }
            }
        }

        const usage = this.recordUsage(inputTokens, outputTokens);

        return {
            text: fullText,
            inputTokens,
            outputTokens,
            costUSD: usage.totalCostUSD
        };
    }
}

function normalizeLMStudioBaseUrl(url: string): string {
    let clean = (url || "").trim().replace(/\/$/, "");
    if (!clean) return "";
    if (!clean.endsWith("/v1")) {
        clean += "/v1";
    }
    return clean;
}

export class LMStudioProvider extends BaseAIProvider {
    readonly providerId = "lmstudio";
    readonly model: string;
    readonly baseUrl: string;

    constructor(baseUrl: string, model: string, customPricing?: Record<string, ModelPricing>) {
        super(customPricing);
        this.baseUrl = normalizeLMStudioBaseUrl(baseUrl);
        this.model = model || "local-model";
    }

    async prompt(history: ChatHistoryItem[], userPrompt: string): Promise<ProviderResponse> {
        const [actualModel] = this.model.split(":");
        const messages: { role: string; content: any }[] = [];

        for (const item of history) {
            const role = item.type === "system" ? "system" : item.type === "model" || item.type === "assistant" ? "assistant" : "user";
            messages.push({ role, content: formatOpenAIMessageContent(item) });
        }

        if (userPrompt) {
            messages.push({ role: "user", content: userPrompt });
        }

        const endpoint = `${this.baseUrl}/chat/completions`;
        const payload = {
            model: actualModel || "local-model",
            messages,
            stream: false
        };

        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`LM Studio API Error (${res.status}): ${errText}`);
        }

        const data = await res.json();
        const choice = data.choices?.[0];
        const text = choice?.message?.content || choice?.text || "";
        const inputTokens = data.usage?.prompt_tokens || 0;
        const outputTokens = data.usage?.completion_tokens || 0;
        const usage = this.recordUsage(inputTokens, outputTokens);

        return {
            text,
            inputTokens,
            outputTokens,
            costUSD: usage.totalCostUSD
        };
    }

    async promptStream(history: ChatHistoryItem[], userPrompt: string, onToken: TokenCallback): Promise<ProviderResponse> {
        const [actualModel] = this.model.split(":");
        const messages: { role: string; content: any }[] = [];

        for (const item of history) {
            const role = item.type === "system" ? "system" : item.type === "model" || item.type === "assistant" ? "assistant" : "user";
            messages.push({ role, content: formatOpenAIMessageContent(item) });
        }

        if (userPrompt) {
            messages.push({ role: "user", content: userPrompt });
        }

        const endpoint = `${this.baseUrl}/chat/completions`;
        const payload = {
            model: actualModel || "local-model",
            messages,
            stream: true
        };

        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`LM Studio Stream API Error (${res.status}): ${errText}`);
        }

        if (!res.body) {
            throw new Error("No response body received from LM Studio stream endpoint.");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullText = "";
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || trimmed.startsWith(":")) continue;

                if (trimmed.startsWith("data: ")) {
                    const dataStr = trimmed.substring(6).trim();
                    if (dataStr === "[DONE]") continue;

                    try {
                        const data = JSON.parse(dataStr);
                        const choice = data.choices?.[0];
                        const token = choice?.delta?.content || choice?.delta?.text || choice?.text || "";
                        if (token) {
                            fullText += token;
                            onToken(token);
                        }
                    } catch {
                        // Ignore partial chunks
                    }
                }
            }
        }

        const usage = this.recordUsage(0, 0);

        return {
            text: fullText,
            inputTokens: 0,
            outputTokens: 0,
            costUSD: usage.totalCostUSD
        };
    }
}

export class CustomOpenAIProvider extends BaseAIProvider {
    readonly providerId = "custom-openai";
    readonly model: string;
    readonly apiKey: string;
    readonly baseUrl: string;

    constructor(baseUrl: string, apiKey: string, model: string, customPricing?: Record<string, ModelPricing>) {
        super(customPricing);
        this.baseUrl = (baseUrl || "http://localhost:1234/v1").trim().replace(/\/$/, "");
        this.apiKey = apiKey || "custom-key";
        this.model = model || "local-model";
    }

    async prompt(history: ChatHistoryItem[], userPrompt: string): Promise<ProviderResponse> {
        const [actualModel] = this.model.split(":");
        const messages: { role: string; content: any }[] = [];

        for (const item of history) {
            const role = item.type === "system" ? "system" : item.type === "model" || item.type === "assistant" ? "assistant" : "user";
            messages.push({ role, content: formatOpenAIMessageContent(item) });
        }

        if (userPrompt) {
            messages.push({ role: "user", content: userPrompt });
        }

        const endpoint = `${this.baseUrl}/chat/completions`;
        const payload = {
            model: actualModel || "local-model",
            messages,
            stream: false
        };

        const headers: Record<string, string> = { "Content-Type": "application/json" };
        if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

        const res = await fetch(endpoint, {
            method: "POST",
            headers,
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Custom OpenAI API Error (${res.status}): ${errText}`);
        }

        const data = await res.json();
        const choice = data.choices?.[0];
        const text = choice?.message?.content || choice?.text || "";
        const inputTokens = data.usage?.prompt_tokens || 0;
        const outputTokens = data.usage?.completion_tokens || 0;
        const usage = this.recordUsage(inputTokens, outputTokens);

        return {
            text,
            inputTokens,
            outputTokens,
            costUSD: usage.totalCostUSD
        };
    }

    async promptStream(history: ChatHistoryItem[], userPrompt: string, onToken: TokenCallback): Promise<ProviderResponse> {
        const [actualModel] = this.model.split(":");
        const messages: { role: string; content: any }[] = [];

        for (const item of history) {
            const role = item.type === "system" ? "system" : item.type === "model" || item.type === "assistant" ? "assistant" : "user";
            messages.push({ role, content: formatOpenAIMessageContent(item) });
        }

        if (userPrompt) {
            messages.push({ role: "user", content: userPrompt });
        }

        const endpoint = `${this.baseUrl}/chat/completions`;
        const payload = {
            model: actualModel || "local-model",
            messages,
            stream: true
        };

        const headers: Record<string, string> = { "Content-Type": "application/json" };
        if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

        const res = await fetch(endpoint, {
            method: "POST",
            headers,
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Custom OpenAI Stream API Error (${res.status}): ${errText}`);
        }

        if (!res.body) {
            throw new Error("No response body received from Custom OpenAI stream endpoint.");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullText = "";
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || trimmed.startsWith(":")) continue;

                if (trimmed.startsWith("data: ")) {
                    const dataStr = trimmed.substring(6).trim();
                    if (dataStr === "[DONE]") continue;

                    try {
                        const data = JSON.parse(dataStr);
                        const choice = data.choices?.[0];
                        const token = choice?.delta?.content || choice?.delta?.text || choice?.text || "";
                        if (token) {
                            fullText += token;
                            onToken(token);
                        }
                    } catch {
                        // Ignore partial chunks
                    }
                }
            }
        }

        const usage = this.recordUsage(0, 0);

        return {
            text: fullText,
            inputTokens: 0,
            outputTokens: 0,
            costUSD: usage.totalCostUSD
        };
    }
}

export class ProviderFactory {
    /**
     * Registers a custom AI service provider implementation via code so any consumer of `@nexustools/aitooling`
     * can plug in its own AI backend into `ProviderFactory`, `MultiProviderToolStreamClient`, and the AI UI.
     */
    static registerCustomProvider(adapter: CustomAiServiceAdapter): void {
        AiProviderRegistry.registerProvider(adapter);
    }

    static unregisterCustomProvider(providerId: string): boolean {
        return AiProviderRegistry.unregisterProvider(providerId);
    }

    static getRegisteredCustomProviders(): CustomAiServiceAdapter[] {
        return AiProviderRegistry.getAllProviders();
    }

    static createProvider(config: AppConfig & Record<string, any>, oauthManager?: GoogleOAuthManager): BaseAIProvider {
        const customAdapter = AiProviderRegistry.getProvider(String(config.activeProvider || ""));
        if (customAdapter?.createProvider) {
            return customAdapter.createProvider(config, oauthManager);
        }

        const pricing = config.customPricing;

        switch (config.activeProvider as string) {
            case "default":
            case "lmstudio": {
                const target = config.lmStudioTarget || { baseUrl: "", model: "" };
                return new LMStudioProvider(target.baseUrl, target.model, pricing);
            }
            case "openai":
                return new OpenAIProvider(config.openai.apiKey, config.openai.model, pricing);
            case "claude":
                return new ClaudeProvider(config.claude.apiKey, config.claude.model, pricing);
            case "gemini":
                return new GeminiProvider(config.gemini, oauthManager, pricing, false);
            case "gemini-googleone":
                return new GeminiProvider(config.gemini, oauthManager, pricing, true);
            case "grok":
                return new GrokProvider(config.grok.apiKey, config.grok.model, pricing);
            case "custom-openai": {
                const target =
                    config.customOpenAITargets?.find(t => t.id === config.customOpenAI.activeTargetId) ||
                    config.customOpenAITargets?.[0] ||
                    { baseUrl: "", apiKey: "", model: "" };
                return new CustomOpenAIProvider(target.baseUrl, target.apiKey, target.model, pricing);
            }
            case "custom-claude": {
                const target =
                    config.customClaudeTargets?.find(t => t.id === config.customClaude.activeTargetId) ||
                    config.customClaudeTargets?.[0] ||
                    { baseUrl: "", apiKey: "", model: "" };
                return new CustomClaudeProvider(target.baseUrl, target.apiKey, target.model, pricing);
            }
            default:
                throw new Error(`Unsupported provider type: ${config.activeProvider}`);
        }
    }
}

// ============================================================================
// MULTI-PROVIDER TOOL-CALLING STREAM CLIENT (FOR UI & AGENT STREAMING)
// ============================================================================

export type AiReasoningLevel = "low" | "medium" | "max";
export type AiReasoningMode = "none" | AiReasoningLevel;

export interface AiChatMessage {
    role: "system" | "user" | "assistant" | "tool";
    content?: string | null;
    name?: string;
    tool_call_id?: string;
    tool_calls?: Array<{
        id: string;
        type: "function";
        function: {
            name: string;
            arguments: string;
        };
    }>;
}

export interface AiStreamRequestPayload {
    provider?: ProviderType;
    model: string;
    messages: AiChatMessage[];
    tools: OpenAiToolSchema[];
    tool_choice: "auto";
    stream: true;
    temperature: number;
    reasoning_effort: "none" | "low" | "medium" | "high";
    reasoning_level?: AiReasoningLevel;
    max_reasoning_tokens?: number;
}

export type CustomAiStreamHandler = (
    payload: AiStreamRequestPayload,
    baseUrl: string,
    onChunk: (chunk: any) => void,
    signal?: AbortSignal
) => Promise<void>;

export interface MultiProviderClientConfig {
    activeProvider?: ProviderType;
    lmStudioUrl?: string;
    aiModel?: string;
    openaiApiKey?: string;
    openaiModel?: string;
    claudeApiKey?: string;
    claudeModel?: string;
    geminiApiKey?: string;
    geminiModel?: string;
    geminiOAuthTokens?: {
        access_token?: string;
        refresh_token?: string;
        expiry_date?: number;
    };
    grokApiKey?: string;
    grokModel?: string;
    customOpenAiBaseUrl?: string;
    customOpenAiApiKey?: string;
    customOpenAiModel?: string;
    customClaudeBaseUrl?: string;
    customClaudeApiKey?: string;
    customClaudeModel?: string;
}

export interface AiStreamClientHostContext {
    configManager?: {
        getConfig(): MultiProviderClientConfig & Record<string, any>;
    };
}

/**
 * Abstract parent class for multi-provider streaming AI clients (`LM Studio`, `OpenAI`, `Anthropic Claude`,
 * `Google Gemini` (API Key & Google One OAuth), `xAI Grok`, `Custom OpenAI`, and `Custom Claude`).
 *
 * Emits unified OpenAI-compatible SSE chunks (`choices[0].delta.content`, `choices[0].delta.reasoning_content`,
 * `choices[0].delta.tool_calls`, `{ done: true }`, `{ error: ... }`) so live UI components (`AiLiveChatView` /
 * `AiAssistantDockPanel`) work identically across every AI service.
 */
export abstract class AbstractAiStreamClient<TContext extends AiStreamClientHostContext = any> {
    public static activeDocumentGetter?: () => Document | undefined;

    protected readonly state: TContext;
    protected readonly urlOverrideGetter?: () => string | null | undefined;
    protected readonly toolsSchemaGetter?: (context: TContext) => OpenAiToolSchema[];
    protected readonly systemPromptBuilder?: (context: TContext, reasoningMode: AiReasoningMode) => string;

    constructor(
        state: TContext,
        urlOverrideGetter?: () => string | null | undefined,
        toolsSchemaGetter?: (context: TContext) => OpenAiToolSchema[],
        systemPromptBuilder?: (context: TContext, reasoningMode: AiReasoningMode) => string
    ) {
        this.state = state;
        this.urlOverrideGetter = urlOverrideGetter;
        this.toolsSchemaGetter = toolsSchemaGetter;
        this.systemPromptBuilder = systemPromptBuilder;
    }

    public getConfig(): MultiProviderClientConfig & Record<string, any> {
        return this.state?.configManager?.getConfig?.() || {};
    }

    public getActiveProvider(): ProviderType {
        const cfg = this.getConfig();
        return (cfg.activeProvider || "lmstudio") as ProviderType;
    }

    public getBaseUrl(): string {
        const directOverride = this.urlOverrideGetter?.();
        if (directOverride) {
            return directOverride.replace(/\/+$/, "");
        }
        const targetDoc =
            AbstractAiStreamClient.activeDocumentGetter?.() ||
            (typeof document !== "undefined" ? document : undefined);
        const overrideUrl = targetDoc
            ?.getElementById("panel-ai-assistant")
            ?.getAttribute("data-lm-studio-url");
        if (overrideUrl) {
            return overrideUrl.replace(/\/+$/, "");
        }
        const cfg = this.getConfig();
        const provider = this.getActiveProvider();
        const customAdapter = AiProviderRegistry.getProvider(String(provider));
        if (customAdapter?.defaultBaseUrl) {
            return (cfg[`${provider}BaseUrl`] || customAdapter.defaultBaseUrl).replace(/\/+$/, "");
        }
        if (provider === "openai") return "https://api.openai.com";
        if (provider === "grok") return "https://api.x.ai";
        if (provider === "claude") return "https://api.anthropic.com";
        if (provider === "gemini" || provider === "gemini-googleone") {
            return "https://generativelanguage.googleapis.com";
        }
        if (provider === "custom-openai") {
            return (cfg.customOpenAiBaseUrl || cfg.lmStudioUrl || "http://localhost:1234").replace(/\/v1\/?$/, "").replace(/\/+$/, "");
        }
        if (provider === "custom-claude") {
            return (cfg.customClaudeBaseUrl || "http://localhost:8080").replace(/\/+$/, "");
        }
        return (cfg.lmStudioUrl || "http://localhost:1234").replace(/\/v1\/?$/, "").replace(/\/+$/, "");
    }

    public getApiKey(): string {
        const cfg = this.getConfig();
        const provider = this.getActiveProvider();
        const customAdapter = AiProviderRegistry.getProvider(String(provider));
        if (customAdapter) {
            return cfg[`${provider}ApiKey`] || cfg.apiKey || "";
        }
        if (provider === "openai") return cfg.openaiApiKey || cfg.openai?.apiKey || "";
        if (provider === "claude") return cfg.claudeApiKey || cfg.claude?.apiKey || "";
        if (provider === "gemini" || provider === "gemini-googleone") return cfg.geminiApiKey || cfg.gemini?.apiKey || "";
        if (provider === "grok") return cfg.grokApiKey || cfg.grok?.apiKey || "";
        if (provider === "custom-openai") return cfg.customOpenAiApiKey || "";
        if (provider === "custom-claude") return cfg.customClaudeApiKey || "";
        return "";
    }

    public getModelName(): string {
        const cfg = this.getConfig();
        const provider = this.getActiveProvider();
        const customAdapter = AiProviderRegistry.getProvider(String(provider));
        if (customAdapter) {
            return cfg[`${provider}Model`] || cfg.aiModel || customAdapter.defaultModel || "custom-model";
        }
        if (provider === "openai") return cfg.openaiModel || cfg.openai?.model || cfg.aiModel || "gpt-4o";
        if (provider === "claude") return cfg.claudeModel || cfg.claude?.model || cfg.aiModel || "claude-3-5-sonnet-20241022";
        if (provider === "gemini" || provider === "gemini-googleone") return cfg.geminiModel || cfg.gemini?.model || cfg.aiModel || "gemini-2.5-flash";
        if (provider === "grok") return cfg.grokModel || cfg.grok?.model || cfg.aiModel || "grok-2";
        if (provider === "custom-openai") return cfg.customOpenAiModel || cfg.aiModel || "local-model";
        if (provider === "custom-claude") return cfg.customClaudeModel || cfg.aiModel || "custom-claude";
        return cfg.aiModel || "local-model";
    }

    public getProviderDisplayName(): string {
        const p = this.getActiveProvider();
        const customAdapter = AiProviderRegistry.getProvider(String(p));
        if (customAdapter) {
            return customAdapter.displayName;
        }
        switch (p) {
            case "openai": return "OpenAI";
            case "claude": return "Anthropic Claude";
            case "gemini": return "Google Gemini";
            case "gemini-googleone": return "Gemini (Google One OAuth)";
            case "grok": return "xAI Grok";
            case "custom-openai": return "Custom OpenAI";
            case "custom-claude": return "Custom Claude";
            case "lmstudio":
            default:
                return "LM Studio";
        }
    }

    public buildSystemPrompt(reasoningMode: AiReasoningMode): string {
        if (this.systemPromptBuilder) {
            return this.systemPromptBuilder(this.state, reasoningMode);
        }
        return `You are a helpful AI Assistant with full tool access. Reasoning mode: ${reasoningMode}.`;
    }

    public getToolsSchema(): OpenAiToolSchema[] {
        return this.toolsSchemaGetter ? this.toolsSchemaGetter(this.state) : [];
    }

    public buildRequestPayload(
        conversationMessages: AiChatMessage[],
        reasoningMode: AiReasoningMode
    ): AiStreamRequestPayload {
        const systemMsg: AiChatMessage = {
            role: "system",
            content: this.buildSystemPrompt(reasoningMode)
        };

        if (reasoningMode === "none") {
            return {
                provider: this.getActiveProvider(),
                model: this.getModelName(),
                messages: [systemMsg, ...conversationMessages],
                tools: this.getToolsSchema(),
                tool_choice: "auto",
                stream: true,
                temperature: 0.3,
                reasoning_effort: "none",
                max_reasoning_tokens: 0
            };
        }

        const mappedEffort: "low" | "medium" | "high" =
            reasoningMode === "max" ? "high" : reasoningMode;

        return {
            provider: this.getActiveProvider(),
            model: this.getModelName(),
            messages: [systemMsg, ...conversationMessages],
            tools: this.getToolsSchema(),
            tool_choice: "auto",
            stream: true,
            temperature: 0.4,
            reasoning_effort: mappedEffort,
            reasoning_level: reasoningMode
        };
    }

    public abstract streamChatCompletion(
        messages: AiChatMessage[],
        reasoningMode: AiReasoningMode,
        onChunk: (chunk: any) => void,
        signal?: AbortSignal
    ): Promise<void>;
}

/**
 * Multi-provider streaming client supporting LM Studio (`http://localhost:1234` default),
 * OpenAI, Anthropic Claude, Google Gemini (API Key & Google One OAuth), xAI Grok,
 * Custom OpenAI, and Custom Claude endpoints with full tool calling and 3 reasoning levels.
 */
export class LmStudioAiClient<TContext extends AiStreamClientHostContext = any> extends AbstractAiStreamClient<TContext> {
    public static customStreamHandler: CustomAiStreamHandler | null = null;
    public static lastRequestPayload: AiStreamRequestPayload | null = null;
    public static lastRequestUrl: string | null = null;

    public async fetchAvailableModels(): Promise<string[]> {
        const provider = this.getActiveProvider();
        const apiKey = this.getApiKey();
        const baseUrl = this.getBaseUrl();
        const customAdapter = AiProviderRegistry.getProvider(String(provider));
        if (customAdapter) {
            if (customAdapter.fetchModels) {
                try {
                    const list = await customAdapter.fetchModels(apiKey, baseUrl, this.getConfig());
                    const mapped = list.map((m) => (typeof m === "string" ? m : m.id)).filter(Boolean);
                    if (mapped.length > 0) return mapped;
                } catch {
                    // Fallback below
                }
            }
            if (customAdapter.fallbackModels && customAdapter.fallbackModels.length > 0) {
                return customAdapter.fallbackModels.map((m) => m.id);
            }
            return [this.getModelName()];
        }

        // For OpenAI, Claude, Gemini, Grok: query ModelCatalog first
        if (provider && provider !== "lmstudio" && provider !== "custom-openai") {
            try {
                const models = await ModelCatalog.fetchAvailableModels(provider, apiKey, baseUrl);
                if (models.length > 0) {
                    return models.map(m => m.id);
                }
            } catch {
                // Fallback below
            }
        }

        // Direct HTTP /v1/models check for LM Studio / OpenAI-compatible servers
        return new Promise((resolve) => {
            try {
                const targetUrl = new URL(`${baseUrl}/v1/models`);
                const client = targetUrl.protocol === "https:" ? https : http;
                const headers: Record<string, string> = {};
                if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

                const req = client.request(
                    targetUrl,
                    { method: "GET", headers, timeout: 2500 },
                    (res) => {
                        let raw = "";
                        res.on("data", chunk => (raw += chunk.toString()));
                        res.on("end", () => {
                            try {
                                const parsed = JSON.parse(raw);
                                if (Array.isArray(parsed?.data)) {
                                    const ids = parsed.data.map((m: any) => String(m.id)).filter(Boolean);
                                    if (ids.length > 0) {
                                        resolve(ids);
                                        return;
                                    }
                                }
                            } catch {
                                // Ignore parse error
                            }
                            const fallbacks = ModelCatalog.getFallbackModels(provider || "lmstudio").map(m => m.id);
                            resolve(fallbacks.length > 0 ? fallbacks : [this.getModelName()]);
                        });
                    }
                );
                req.on("error", () => {
                    const fallbacks = ModelCatalog.getFallbackModels(provider || "lmstudio").map(m => m.id);
                    resolve(fallbacks.length > 0 ? fallbacks : [this.getModelName()]);
                });
                req.on("timeout", () => {
                    req.destroy();
                    resolve([this.getModelName()]);
                });
                req.end();
            } catch {
                resolve([this.getModelName()]);
            }
        });
    }

    public override async streamChatCompletion(
        messages: AiChatMessage[],
        reasoningMode: AiReasoningMode,
        onChunk: (chunk: any) => void,
        signal?: AbortSignal
    ): Promise<void> {
        const provider = this.getActiveProvider();
        const baseUrl = this.getBaseUrl();
        const payload = this.buildRequestPayload(messages, reasoningMode);
        LmStudioAiClient.lastRequestPayload = payload;

        const hasDirectUrlOverride = Boolean(
            this.urlOverrideGetter?.() ||
            (AbstractAiStreamClient.activeDocumentGetter?.() || (typeof document !== "undefined" ? document : undefined))
                ?.getElementById("panel-ai-assistant")
                ?.getAttribute("data-lm-studio-url")
        );

        if (LmStudioAiClient.customStreamHandler) {
            LmStudioAiClient.lastRequestUrl = `${baseUrl}/v1/chat/completions`;
            await LmStudioAiClient.customStreamHandler(payload, baseUrl, onChunk, signal);
            return;
        }

        const customAdapter = AiProviderRegistry.getProvider(String(provider));
        if (customAdapter?.streamToolChatCompletion && !hasDirectUrlOverride) {
            LmStudioAiClient.lastRequestUrl = `${baseUrl}/custom/${customAdapter.providerId}`;
            await customAdapter.streamToolChatCompletion(payload, this.getConfig(), onChunk, signal);
            return;
        }

        // If a test or user explicitly set a local URL override on the panel, or provider is OpenAI-compatible
        // ("lmstudio", "openai", "grok", "custom-openai"), stream via `/v1/chat/completions` SSE
        if (
            hasDirectUrlOverride ||
            provider === "lmstudio" ||
            provider === "openai" ||
            provider === "grok" ||
            provider === "custom-openai" ||
            !provider
        ) {
            await this.streamOpenAiCompatible(payload, baseUrl, onChunk, signal);
            return;
        }

        if (provider === "claude" || provider === "custom-claude") {
            await this.streamClaudeCompatible(payload, baseUrl, onChunk, signal);
            return;
        }

        if (provider === "gemini" || provider === "gemini-googleone") {
            await this.streamGeminiCompatible(payload, baseUrl, onChunk, signal);
            return;
        }

        await this.streamOpenAiCompatible(payload, baseUrl, onChunk, signal);
    }

    protected async streamOpenAiCompatible(
        payload: AiStreamRequestPayload,
        baseUrl: string,
        onChunk: (chunk: any) => void,
        signal?: AbortSignal
    ): Promise<void> {
        const endpoint = `${baseUrl}/v1/chat/completions`;
        LmStudioAiClient.lastRequestUrl = endpoint;
        const apiKey = this.getApiKey();
        const providerLabel = this.getProviderDisplayName();

        await new Promise<void>((resolve) => {
            try {
                const targetUrl = new URL(endpoint);
                const bodyStr = JSON.stringify(payload);
                const client = targetUrl.protocol === "https:" ? https : http;
                const headers: Record<string, string | number> = {
                    "Content-Type": "application/json",
                    "Content-Length": Buffer.byteLength(bodyStr, "utf8")
                };
                if (apiKey) {
                    headers["Authorization"] = `Bearer ${apiKey}`;
                }

                const req = client.request(
                    targetUrl,
                    {
                        method: "POST",
                        headers
                    },
                    (res) => {
                        if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
                            let errBody = "";
                            res.on("data", d => (errBody += d.toString()));
                            res.on("end", () => {
                                onChunk({
                                    error: `${providerLabel} returned HTTP ${res.statusCode}${errBody ? `: ${errBody.slice(0, 160)}` : ""}`
                                });
                                resolve();
                            });
                            return;
                        }

                        let buffer = "";
                        res.on("data", (chunk: Buffer) => {
                            buffer += chunk.toString("utf8");
                            const lines = buffer.split("\n");
                            buffer = lines.pop() || "";

                            for (const line of lines) {
                                const trimmed = line.trim();
                                if (!trimmed || !trimmed.startsWith("data:")) continue;
                                const dataStr = trimmed.slice(5).trim();
                                if (dataStr === "[DONE]") {
                                    onChunk({ done: true });
                                    resolve();
                                    return;
                                }
                                try {
                                    const parsed = JSON.parse(dataStr);
                                    onChunk(parsed);
                                } catch {
                                    // Ignore partial JSON lines
                                }
                            }
                        });

                        res.on("end", () => {
                            if (buffer.trim().startsWith("data:")) {
                                const dataStr = buffer.trim().slice(5).trim();
                                if (dataStr && dataStr !== "[DONE]") {
                                    try {
                                        onChunk(JSON.parse(dataStr));
                                    } catch {
                                        // Ignore
                                    }
                                }
                            }
                            onChunk({ done: true });
                            resolve();
                        });

                        res.on("error", (err) => {
                            if (signal?.aborted) {
                                resolve();
                                return;
                            }
                            onChunk({ error: `${providerLabel} stream error: ${err.message}` });
                            resolve();
                        });
                    }
                );

                if (signal) {
                    if (signal.aborted) {
                        req.destroy();
                        resolve();
                        return;
                    }
                    signal.addEventListener(
                        "abort",
                        () => {
                            req.destroy();
                            resolve();
                        },
                        { once: true }
                    );
                }

                req.on("error", (err) => {
                    if (signal?.aborted) {
                        resolve();
                        return;
                    }
                    onChunk({
                        error: `Cannot connect to ${providerLabel} at ${baseUrl} (${err.message}).`
                    });
                    resolve();
                });

                req.write(bodyStr);
                req.end();
            } catch (err: any) {
                onChunk({ error: err?.message || `Failed to connect to ${providerLabel}` });
                resolve();
            }
        });
    }

    protected async streamClaudeCompatible(
        payload: AiStreamRequestPayload,
        baseUrl: string,
        onChunk: (chunk: any) => void,
        signal?: AbortSignal
    ): Promise<void> {
        const endpoint = `${baseUrl}/v1/messages`;
        LmStudioAiClient.lastRequestUrl = endpoint;
        const apiKey = this.getApiKey();
        if (!apiKey) {
            onChunk({ error: "Anthropic Claude API Key is not configured. Open AI Settings (⚙️) to set your Claude API key." });
            return;
        }

        try {
            const client = new Anthropic({
                apiKey,
                baseURL: baseUrl,
                dangerouslyAllowBrowser: true
            });

            let systemPrompt = "";
            const anthropicMessages: any[] = [];
            for (const msg of payload.messages) {
                if (msg.role === "system") {
                    systemPrompt += (systemPrompt ? "\n\n" : "") + (msg.content || "");
                } else if (msg.role === "tool") {
                    anthropicMessages.push({
                        role: "user",
                        content: [
                            {
                                type: "tool_result",
                                tool_use_id: msg.tool_call_id || "tool_call_0",
                                content: msg.content || "{}"
                            }
                        ]
                    });
                } else if (msg.role === "assistant" && msg.tool_calls && msg.tool_calls.length > 0) {
                    const blocks: any[] = [];
                    if (msg.content) {
                        blocks.push({ type: "text", text: msg.content });
                    }
                    for (const tc of msg.tool_calls) {
                        let parsedInput = {};
                        try {
                            parsedInput = JSON.parse(tc.function.arguments || "{}");
                        } catch {
                            parsedInput = {};
                        }
                        blocks.push({
                            type: "tool_use",
                            id: tc.id,
                            name: tc.function.name,
                            input: parsedInput
                        });
                    }
                    anthropicMessages.push({ role: "assistant", content: blocks });
                } else {
                    anthropicMessages.push({
                        role: msg.role === "assistant" ? "assistant" : "user",
                        content: msg.content || ""
                    });
                }
            }

            const anthropicTools = payload.tools.map((t) => ({
                name: t.function.name,
                description: t.function.description,
                input_schema: t.function.parameters as any
            }));

            const reqBody: any = {
                model: payload.model.split(":")[0],
                max_tokens: 8192,
                system: systemPrompt || undefined,
                messages: anthropicMessages,
                tools: anthropicTools.length > 0 ? anthropicTools : undefined,
                stream: true
            };

            if (payload.reasoning_effort !== "none") {
                const budgetMap: Record<string, number> = { low: 1024, medium: 2048, high: 4096 };
                reqBody.thinking = {
                    type: "enabled",
                    budget_tokens: budgetMap[payload.reasoning_effort] || 2048
                };
            }

            const stream = await client.messages.stream(reqBody);
            let toolCallIndex = -1;

            for await (const event of stream) {
                if (signal?.aborted) break;
                const ev = event as any;
                if (ev.type === "content_block_start") {
                    if (ev.content_block?.type === "tool_use") {
                        toolCallIndex++;
                        onChunk({
                            choices: [
                                {
                                    delta: {
                                        tool_calls: [
                                            {
                                                index: toolCallIndex,
                                                id: ev.content_block.id,
                                                function: {
                                                    name: ev.content_block.name,
                                                    arguments: ""
                                                }
                                            }
                                        ]
                                    }
                                }
                            ]
                        });
                    }
                } else if (ev.type === "content_block_delta") {
                    if (ev.delta?.type === "text_delta" && ev.delta.text) {
                        onChunk({
                            choices: [{ delta: { content: ev.delta.text } }]
                        });
                    } else if (ev.delta?.type === "thinking_delta" && ev.delta.thinking) {
                        onChunk({
                            choices: [{ delta: { reasoning_content: ev.delta.thinking } }]
                        });
                    } else if (ev.delta?.type === "input_json_delta" && ev.delta.partial_json) {
                        onChunk({
                            choices: [
                                {
                                    delta: {
                                        tool_calls: [
                                            {
                                                index: Math.max(0, toolCallIndex),
                                                function: {
                                                    arguments: ev.delta.partial_json
                                                }
                                            }
                                        ]
                                    }
                                }
                            ]
                        });
                    }
                }
            }
            onChunk({ done: true });
        } catch (err: any) {
            onChunk({ error: `Claude error: ${err?.message || String(err)}` });
        }
    }

    protected async streamGeminiCompatible(
        payload: AiStreamRequestPayload,
        baseUrl: string,
        onChunk: (chunk: any) => void,
        signal?: AbortSignal
    ): Promise<void> {
        const modelId = payload.model.split(":")[0] || "gemini-2.5-flash";
        const apiKey = this.getApiKey();
        const cfg = this.getConfig();
        const oauthToken = cfg.geminiOAuthTokens?.access_token || cfg.gemini?.oauthTokens?.access_token;

        const endpoint = `${baseUrl}/v1beta/models/${encodeURIComponent(modelId)}:streamGenerateContent?alt=sse${apiKey && !oauthToken ? `&key=${encodeURIComponent(apiKey)}` : ""}`;
        LmStudioAiClient.lastRequestUrl = endpoint;

        if (!apiKey && !oauthToken) {
            onChunk({ error: "Google Gemini API Key or OAuth sign-in is not configured. Open AI Settings (⚙️) to configure Gemini." });
            return;
        }

        try {
            let systemInstruction = "";
            const contents: any[] = [];
            for (const msg of payload.messages) {
                if (msg.role === "system") {
                    systemInstruction += (systemInstruction ? "\n\n" : "") + (msg.content || "");
                } else if (msg.role === "tool") {
                    let parsedResp = {};
                    try {
                        parsedResp = JSON.parse(msg.content || "{}");
                    } catch {
                        parsedResp = { result: msg.content };
                    }
                    contents.push({
                        role: "function",
                        parts: [
                            {
                                functionResponse: {
                                    name: msg.name || "tool",
                                    response: parsedResp
                                }
                            }
                        ]
                    });
                } else if (msg.role === "assistant" && msg.tool_calls && msg.tool_calls.length > 0) {
                    const parts: any[] = [];
                    if (msg.content) parts.push({ text: msg.content });
                    for (const tc of msg.tool_calls) {
                        let argsObj = {};
                        try {
                            argsObj = JSON.parse(tc.function.arguments || "{}");
                        } catch {
                            argsObj = {};
                        }
                        parts.push({
                            functionCall: {
                                name: tc.function.name,
                                args: argsObj
                            }
                        });
                    }
                    contents.push({ role: "model", parts });
                } else {
                    contents.push({
                        role: msg.role === "assistant" ? "model" : "user",
                        parts: [{ text: msg.content || "" }]
                    });
                }
            }

            const geminiBody: any = {
                contents,
                ...(systemInstruction ? { systemInstruction: { parts: [{ text: systemInstruction }] } } : {}),
                ...(payload.tools.length > 0
                    ? {
                          tools: [
                              {
                                  functionDeclarations: payload.tools.map((t) => ({
                                      name: t.function.name,
                                      description: t.function.description,
                                      parameters: t.function.parameters
                                  }))
                              }
                          ]
                      }
                    : {})
            };

            const headers: Record<string, string> = { "Content-Type": "application/json" };
            if (oauthToken) {
                headers["Authorization"] = `Bearer ${oauthToken}`;
            }

            const res = await fetch(endpoint, {
                method: "POST",
                headers,
                body: JSON.stringify(geminiBody),
                signal
            });

            if (!res.ok || !res.body) {
                const errText = await res.text();
                onChunk({ error: `Gemini HTTP ${res.status}: ${errText.slice(0, 160)}` });
                return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";
            let toolCallIndex = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed.startsWith("data:")) continue;
                    const dataStr = trimmed.slice(5).trim();
                    if (!dataStr || dataStr === "[DONE]") continue;
                    try {
                        const parsed = JSON.parse(dataStr);
                        const parts = parsed.candidates?.[0]?.content?.parts || [];
                        for (const p of parts) {
                            if (p.thought && p.text) {
                                onChunk({ choices: [{ delta: { reasoning_content: p.text } }] });
                            } else if (p.text) {
                                onChunk({ choices: [{ delta: { content: p.text } }] });
                            } else if (p.functionCall) {
                                const idx = toolCallIndex++;
                                onChunk({
                                    choices: [
                                        {
                                            delta: {
                                                tool_calls: [
                                                    {
                                                        index: idx,
                                                        id: `call_gemini_${idx}`,
                                                        function: {
                                                            name: p.functionCall.name,
                                                            arguments: JSON.stringify(p.functionCall.args || {})
                                                        }
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                });
                            }
                        }
                    } catch {
                        // Ignore partial JSON
                    }
                }
            }
            onChunk({ done: true });
        } catch (err: any) {
            if (!signal?.aborted) {
                onChunk({ error: `Gemini stream error: ${err?.message || String(err)}` });
            }
        }
    }
}

export { LmStudioAiClient as MultiProviderToolStreamClient };
