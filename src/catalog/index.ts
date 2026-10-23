import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

export interface ModelThinkingConfig {
    parameterName: "reasoning.effort" | "thinking.type" | "thinkingConfig.thinkingLevel";
    supportedValues: string[];
    default?: string;
}

export interface ModelInfo {
    id: string;
    name: string;
    contextWindow: number;
    maxOutputTokens?: number;
    inputCostPer1M: number;
    outputCostPer1M: number;
    cachedInputCostPer1M?: number;
    thinking?: ModelThinkingConfig;
    description?: string;
    parameters?: string;
}

export const defaultModelDatabase: Record<string, ModelInfo[]> = {
    openai: [
        {
            id: "gpt-6-astra",
            name: "GPT-6 Astra (1.05M Context | $10.00/$50.00 MTok)",
            contextWindow: 1050000,
            maxOutputTokens: 128000,
            inputCostPer1M: 10.00,
            outputCostPer1M: 50.00,
            thinking: {
                parameterName: "reasoning.effort",
                supportedValues: ["low", "medium", "high", "xhigh", "max"],
                default: "medium"
            },
            description: "Ultra-high intelligence frontier model."
        },
        {
            id: "gpt-6.1-sol",
            name: "GPT-6.1 Sol (1.05M Context | $2.00/$10.00 MTok)",
            contextWindow: 1050000,
            maxOutputTokens: 128000,
            inputCostPer1M: 2.00,
            outputCostPer1M: 10.00,
            thinking: {
                parameterName: "reasoning.effort",
                supportedValues: ["low", "medium", "high", "xhigh", "max"],
                default: "medium"
            },
            description: "Balanced frontier reasoning and execution model."
        },
        {
            id: "gpt-6-luna",
            name: "GPT-6 Luna (1.05M Context | $0.10/$0.50 MTok)",
            contextWindow: 1050000,
            maxOutputTokens: 128000,
            inputCostPer1M: 0.10,
            outputCostPer1M: 0.50,
            thinking: {
                parameterName: "reasoning.effort",
                supportedValues: ["none", "low", "medium", "high", "xhigh", "max"],
                default: "none"
            },
            description: "High-speed, low-cost frontier model."
        },
        {
            id: "gpt-4o",
            name: "GPT-4o (128k Context | $2.50/$10.00 MTok)",
            contextWindow: 128000,
            maxOutputTokens: 16384,
            inputCostPer1M: 2.50,
            outputCostPer1M: 10.00,
            description: "Multimodal flagship model."
        },
        {
            id: "gpt-4o-mini",
            name: "GPT-4o Mini (128k Context | $0.15/$0.60 MTok)",
            contextWindow: 128000,
            maxOutputTokens: 16384,
            inputCostPer1M: 0.15,
            outputCostPer1M: 0.60,
            description: "Fast, economical multimodal model."
        },
        {
            id: "o3-mini",
            name: "o3-mini (200k Context | $1.10/$4.40 MTok)",
            contextWindow: 200000,
            maxOutputTokens: 100000,
            inputCostPer1M: 1.10,
            outputCostPer1M: 4.40,
            thinking: {
                parameterName: "reasoning.effort",
                supportedValues: ["low", "medium", "high"],
                default: "medium"
            },
            description: "Reasoning model for math, coding, and science."
        },
        {
            id: "o1",
            name: "o1 (200k Context | $15.00/$60.00 MTok)",
            contextWindow: 200000,
            maxOutputTokens: 100000,
            inputCostPer1M: 15.00,
            outputCostPer1M: 60.00,
            thinking: {
                parameterName: "reasoning.effort",
                supportedValues: ["low", "medium", "high"],
                default: "medium"
            },
            description: "Deep reasoning model for high complexity."
        }
    ],
    claude: [
        {
            id: "claude-opus-4-8",
            name: "Claude Opus 4.8 (200k Context | $15.00/$75.00 MTok)",
            contextWindow: 200000,
            maxOutputTokens: 128000,
            inputCostPer1M: 15.00,
            outputCostPer1M: 75.00,
            thinking: {
                parameterName: "thinking.type",
                supportedValues: ["adaptive", "low", "medium", "high"],
                default: "adaptive"
            },
            description: "Deep analytical and research reasoning model."
        },
        {
            id: "claude-sonnet-5",
            name: "Claude Sonnet 5 (200k Context | $3.00/$15.00 MTok)",
            contextWindow: 200000,
            maxOutputTokens: 128000,
            inputCostPer1M: 3.00,
            outputCostPer1M: 15.00,
            thinking: {
                parameterName: "thinking.type",
                supportedValues: ["adaptive", "low", "medium", "high"],
                default: "adaptive"
            },
            description: "Next-gen flagship software engineering model."
        },
        {
            id: "claude-haiku-5-5",
            name: "Claude Haiku 5.5 (1M Context | $0.10/$0.50 MTok)",
            contextWindow: 1000000,
            maxOutputTokens: 128000,
            inputCostPer1M: 0.10,
            outputCostPer1M: 0.50,
            thinking: {
                parameterName: "thinking.type",
                supportedValues: ["adaptive", "low", "medium", "high"],
                default: "adaptive"
            },
            description: "Ultra-fast 1M context model with adaptive thinking."
        },
        {
            id: "claude-3-7-sonnet-latest",
            name: "Claude 3.7 Sonnet (200k Context | $3.00/$15.00 MTok)",
            contextWindow: 200000,
            maxOutputTokens: 128000,
            inputCostPer1M: 3.00,
            outputCostPer1M: 15.00,
            thinking: {
                parameterName: "thinking.type",
                supportedValues: ["adaptive", "low", "medium", "high"],
                default: "adaptive"
            },
            description: "Hybrid reasoning model with extended thinking budget."
        },
        {
            id: "claude-3-5-sonnet-20241022",
            name: "Claude 3.5 Sonnet (200k Context | $3.00/$15.00 MTok)",
            contextWindow: 200000,
            maxOutputTokens: 8192,
            inputCostPer1M: 3.00,
            outputCostPer1M: 15.00,
            description: "Industry standard for coding and general tasks."
        }
    ],
    gemini: [
        {
            id: "gemini-3.8-flash",
            name: "Gemini 3.8 Flash (1.05M Context | $0.75/$3.75 MTok)",
            contextWindow: 1048576,
            maxOutputTokens: 65536,
            inputCostPer1M: 0.75,
            outputCostPer1M: 3.75,
            thinking: {
                parameterName: "thinkingConfig.thinkingLevel",
                supportedValues: ["low", "medium", "high"],
                default: "high"
            },
            description: "Next-gen multimodal reasoning Flash model."
        },
        {
            id: "gemini-3.7-flash",
            name: "Gemini 3.7 Flash (1.05M Context | $0.75/$3.75 MTok)",
            contextWindow: 1048576,
            maxOutputTokens: 65536,
            inputCostPer1M: 0.75,
            outputCostPer1M: 3.75,
            thinking: {
                parameterName: "thinkingConfig.thinkingLevel",
                supportedValues: ["low", "medium", "high"],
                default: "medium"
            },
            description: "Multimodal Flash model with thinking level controls."
        },
        {
            id: "gemini-3.6-flash",
            name: "Gemini 3.6 Flash (1.05M Context | $0.75/$3.75 MTok)",
            contextWindow: 1048576,
            maxOutputTokens: 65536,
            inputCostPer1M: 0.75,
            outputCostPer1M: 3.75,
            thinking: {
                parameterName: "thinkingConfig.thinkingLevel",
                supportedValues: ["low", "medium", "high"],
                default: "medium"
            },
            description: "Fast reasoning Flash model."
        },
        {
            id: "gemini-3.5-flash-lite",
            name: "Gemini 3.5 Flash-Lite (1.05M Context | $0.25/$1.25 MTok)",
            contextWindow: 1048576,
            maxOutputTokens: 65536,
            inputCostPer1M: 0.25,
            outputCostPer1M: 1.25,
            thinking: {
                parameterName: "thinkingConfig.thinkingLevel",
                supportedValues: ["low", "medium"],
                default: "low"
            },
            description: "Lightweight, ultra-fast Flash model."
        },
        {
            id: "gemini-2.5-flash",
            name: "Gemini 2.5 Flash (1.05M Context | $0.075/$0.30 MTok)",
            contextWindow: 1048576,
            maxOutputTokens: 65536,
            inputCostPer1M: 0.075,
            outputCostPer1M: 0.30,
            description: "1M Context high-speed model."
        },
        {
            id: "gemini-2.5-pro",
            name: "Gemini 2.5 Pro (2.1M Context | $1.25/$5.00 MTok)",
            contextWindow: 2097152,
            maxOutputTokens: 65536,
            inputCostPer1M: 1.25,
            outputCostPer1M: 5.00,
            thinking: {
                parameterName: "thinkingConfig.thinkingLevel",
                supportedValues: ["low", "medium", "high"],
                default: "high"
            },
            description: "2M Context reasoning model."
        }
    ],
    "gemini-googleone": [
        {
            id: "gemini-3.6-flash",
            name: "Gemini 3.6 Flash (Google One Subscription | $0.00001 MTok)",
            contextWindow: 1048576,
            maxOutputTokens: 65536,
            inputCostPer1M: 0.00001,
            outputCostPer1M: 0.00001,
            thinking: {
                parameterName: "thinkingConfig.thinkingLevel",
                supportedValues: ["low", "medium", "high"],
                default: "medium"
            },
            description: "Included with Google One subscription for AI Code APIs."
        },
        {
            id: "gemini-3.5-flash-lite",
            name: "Gemini 3.5 Flash-Lite (Google One Subscription | $0.00001 MTok)",
            contextWindow: 1048576,
            maxOutputTokens: 65536,
            inputCostPer1M: 0.00001,
            outputCostPer1M: 0.00001,
            thinking: {
                parameterName: "thinkingConfig.thinkingLevel",
                supportedValues: ["low", "medium"],
                default: "low"
            },
            description: "Included lightweight model with Google One subscription."
        },
        {
            id: "gemini-3.1-pro-preview",
            name: "Gemini 3.1 Pro Preview (Google One Subscription | $0.00001 MTok)",
            contextWindow: 2097152,
            maxOutputTokens: 65536,
            inputCostPer1M: 0.00001,
            outputCostPer1M: 0.00001,
            thinking: {
                parameterName: "thinkingConfig.thinkingLevel",
                supportedValues: ["low", "medium", "high"],
                default: "high"
            },
            description: "High-reasoning Pro model included with Google One subscription."
        }
    ],
    grok: [
        {
            id: "grok-4.7",
            name: "Grok 4.7 (500k Context | $2.00/$6.00 MTok | $0.50 Cached)",
            contextWindow: 500000,
            maxOutputTokens: 65536,
            inputCostPer1M: 2.00,
            outputCostPer1M: 6.00,
            cachedInputCostPer1M: 0.50,
            thinking: {
                parameterName: "reasoning.effort",
                supportedValues: ["low", "medium", "high", "xhigh"],
                default: "high"
            },
            description: "xAI 500k context flagship reasoning model."
        },
        {
            id: "grok-2",
            name: "Grok-2 (128k Context | $2.00/$10.00 MTok)",
            contextWindow: 128000,
            maxOutputTokens: 16384,
            inputCostPer1M: 2.00,
            outputCostPer1M: 10.00,
            description: "xAI flagship intelligence model."
        }
    ],
    "custom-openai": [
        {
            id: "local-model",
            name: "Local Model (LM Studio / Ollama / LocalAI)",
            contextWindow: 32768,
            inputCostPer1M: 0.00,
            outputCostPer1M: 0.00,
            description: "Locally running OpenAI-compatible model."
        }
    ],
    "custom-claude": [
        {
            id: "custom-claude",
            name: "Custom Claude Model",
            contextWindow: 100000,
            inputCostPer1M: 0.00,
            outputCostPer1M: 0.00,
            description: "Custom Anthropic-compatible target."
        }
    ],
    lmstudio: [
        {
            id: "local-model",
            name: "Local Model (LM Studio)",
            contextWindow: 32768,
            inputCostPer1M: 0.00,
            outputCostPer1M: 0.00,
            description: "Locally running model via LM Studio."
        }
    ]
};

export class ModelCatalog {
    static getFallbackModels(provider: string): ModelInfo[] {
        return defaultModelDatabase[provider] || [
            {
                id: "default-model",
                name: "Default Model",
                contextWindow: 32768,
                inputCostPer1M: 0,
                outputCostPer1M: 0
            }
        ];
    }

    static async fetchAvailableModels(
        provider: string,
        apiKey?: string,
        baseUrl?: string
    ): Promise<ModelInfo[]> {
        const fallbacks = ModelCatalog.getFallbackModels(provider);

        try {
            if (provider === "openai" && apiKey) {
                const client = new OpenAI({ apiKey, dangerouslyAllowBrowser: true });
                const list = await client.models.list();
                const fetchedMap = new Map<string, ModelInfo>();

                for (const item of fallbacks) {
                    fetchedMap.set(item.id, item);
                }

                for await (const m of list) {
                    if (m.id.startsWith("gpt-") || m.id.startsWith("o1") || m.id.startsWith("o3") || m.id.startsWith("o4")) {
                        if (!fetchedMap.has(m.id)) {
                            fetchedMap.set(m.id, {
                                id: m.id,
                                name: `${m.id} (Discovered)`,
                                contextWindow: 1050000,
                                inputCostPer1M: 2.00,
                                outputCostPer1M: 10.00,
                                thinking: m.id.startsWith("o") || m.id.includes("sol") || m.id.includes("astra") ? {
                                    parameterName: "reasoning.effort",
                                    supportedValues: ["low", "medium", "high", "xhigh"],
                                    default: "medium"
                                } : undefined
                            });
                        }
                    }
                }
                return Array.from(fetchedMap.values());
            }

            if (provider === "claude" && apiKey) {
                try {
                    const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
                    const page = await client.models.list();
                    if (page && page.data && page.data.length > 0) {
                        const fetchedMap = new Map<string, ModelInfo>();
                        for (const item of fallbacks) fetchedMap.set(item.id, item);

                        for (const m of page.data) {
                            if (!fetchedMap.has(m.id)) {
                                fetchedMap.set(m.id, {
                                    id: m.id,
                                    name: m.display_name || m.id,
                                    contextWindow: 200000,
                                    inputCostPer1M: 3.00,
                                    outputCostPer1M: 15.00,
                                    thinking: m.id.includes("3-7") || m.id.includes("5") ? {
                                        parameterName: "thinking.type",
                                        supportedValues: ["adaptive", "low", "medium", "high"],
                                        default: "adaptive"
                                    } : undefined
                                });
                            }
                        }
                        return Array.from(fetchedMap.values());
                    }
                } catch {
                    return fallbacks;
                }
            }

            if (provider === "gemini" && apiKey) {
                const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`;
                const res = await fetch(url);
                if (res.ok) {
                    const data = await res.json();
                    if (data.models && Array.isArray(data.models)) {
                        const fetchedMap = new Map<string, ModelInfo>();
                        for (const item of fallbacks) fetchedMap.set(item.id, item);

                        for (const m of data.models) {
                            if (m.name && m.supportedGenerationMethods?.includes("generateContent")) {
                                const id = m.name.replace(/^models\//, "");
                                if (!fetchedMap.has(id)) {
                                    fetchedMap.set(id, {
                                        id,
                                        name: m.displayName || id,
                                        contextWindow: m.inputTokenLimit || 1048576,
                                        inputCostPer1M: 0.75,
                                        outputCostPer1M: 3.75,
                                        thinking: id.includes("3.") || id.includes("pro") ? {
                                            parameterName: "thinkingConfig.thinkingLevel",
                                            supportedValues: ["low", "medium", "high"],
                                            default: "high"
                                        } : undefined
                                    });
                                }
                            }
                        }
                        return Array.from(fetchedMap.values());
                    }
                }
            }

            if (provider === "grok" && apiKey) {
                const client = new OpenAI({ apiKey, baseURL: "https://api.x.ai/v1", dangerouslyAllowBrowser: true });
                const list = await client.models.list();
                const fetchedMap = new Map<string, ModelInfo>();
                for (const item of fallbacks) fetchedMap.set(item.id, item);

                for await (const m of list) {
                    if (!fetchedMap.has(m.id)) {
                        fetchedMap.set(m.id, {
                            id: m.id,
                            name: `${m.id} (xAI Discovered)`,
                            contextWindow: 500000,
                            inputCostPer1M: 2.00,
                            outputCostPer1M: 6.00,
                            thinking: m.id.includes("4") ? {
                                parameterName: "reasoning.effort",
                                supportedValues: ["low", "medium", "high", "xhigh"],
                                default: "high"
                            } : undefined
                        });
                    }
                }
                return Array.from(fetchedMap.values());
            }

            if (provider === "custom-openai" || provider === "lmstudio") {
                let cleanUrl = (baseUrl || "").trim().replace(/\/$/, "");
                if (provider === "lmstudio") {
                    if (!cleanUrl) return [];
                    if (!cleanUrl.endsWith("/v1")) {
                        cleanUrl += "/v1";
                    }
                }
                if (!cleanUrl) return [];
                const endpoint = cleanUrl + "/models";
                const res = await fetch(endpoint, {
                    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {}
                });
                if (res.ok) {
                    const data = await res.json();
                    if (data.data && Array.isArray(data.data)) {
                        return data.data.map((m: any) => ({
                            id: m.id,
                            name: `${m.id} (${provider === "lmstudio" ? "LM Studio" : "Local Endpoint"})`,
                            contextWindow: m.context_length || m.contextWindow || 16384,
                            inputCostPer1M: 0,
                            outputCostPer1M: 0,
                            parameters: m.params || undefined,
                            description: m.description || undefined
                        }));
                    }
                }
                return [];
            }
        } catch (err) {
            console.warn(`Failed to dynamically fetch models for ${provider}:`, err);
        }

        return fallbacks;
    }
}
