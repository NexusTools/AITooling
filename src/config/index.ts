import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";

export type ProviderType =
    | "openai"
    | "claude"
    | "gemini"
    | "gemini-googleone"
    | "grok"
    | "custom-openai"
    | "custom-claude"
    | "lmstudio"
    | "";

export interface OpenAIConfig {
    apiKey: string;
    model: string;
    thinkingBudgetTokens?: number;
}

export interface ClaudeConfig {
    apiKey: string;
    model: string;
    thinkingBudgetTokens?: number;
}

export interface GeminiConfig {
    apiKey: string;
    model: string;
    oauthTokens?: {
        access_token?: string;
        refresh_token?: string;
        expiry_date?: number;
    };
}

export interface GrokConfig {
    apiKey: string;
    model: string;
}

export interface CustomTargetEntry {
    id: string;
    label: string;
    baseUrl: string;
    apiKey: string;
    model: string;
    contextWindow?: number;
    inputCostPer1M?: number;
    outputCostPer1M?: number;
}

export interface ModelPricing {
    inputCostPer1M: number;
    outputCostPer1M: number;
}

export interface SavedProjectSession {
    id: string;
    title: string;
    folderPath: string;
    isSSH: boolean;
    sshConfig?: {
        host: string;
        port: number;
        username: string;
        auth: string;
        remotePath: string;
    };
}

export interface AppConfig {
    activeProvider: ProviderType;
    openai: OpenAIConfig;
    claude: ClaudeConfig;
    gemini: GeminiConfig;
    grok: GrokConfig;
    customOpenAI: {
        activeTargetId: string;
    };
    customClaude: {
        activeTargetId: string;
    };
    customOpenAITargets: CustomTargetEntry[];
    customClaudeTargets: CustomTargetEntry[];
    lmStudioTarget: CustomTargetEntry;
    lmStudioModelContextSizes: Record<string, number>;
    customPricing: Record<string, ModelPricing>;
    openProjects: SavedProjectSession[];
}

/**
 * Returns the standard OS user configuration directory.
 */
export function getDefaultOSConfigDir(appName = "AICore"): string {
    if (process.platform === "win32") {
        return path.join(process.env.APPDATA || path.join(process.env.USERPROFILE || "", "AppData", "Roaming"), appName);
    } else if (process.platform === "darwin") {
        return path.join(process.env.HOME || "", "Library", "Application Support", appName);
    } else {
        return path.join(process.env.XDG_CONFIG_HOME || path.join(process.env.HOME || "", ".config"), appName.toLowerCase());
    }
}

const defaultConfig: AppConfig = {
    activeProvider: "",
    openai: {
        apiKey: "",
        model: "gpt-4o",
        thinkingBudgetTokens: 2048
    },
    claude: {
        apiKey: "",
        model: "claude-3-5-sonnet-20241022",
        thinkingBudgetTokens: 2048
    },
    gemini: {
        apiKey: "",
        model: "gemini-2.5-flash"
    },
    grok: {
        apiKey: "",
        model: "grok-2"
    },
    customOpenAI: {
        activeTargetId: ""
    },
    customClaude: {
        activeTargetId: ""
    },
    customOpenAITargets: [],
    customClaudeTargets: [],
    lmStudioTarget: {
        id: "",
        label: "",
        baseUrl: "",
        apiKey: "",
        model: "",
        contextWindow: 16384,
        inputCostPer1M: 0,
        outputCostPer1M: 0
    },
    lmStudioModelContextSizes: {},
    customPricing: {},
    openProjects: []
};

export class ConfigManager {
    private readonly configDir: string;
    private readonly configPath: string;
    private config: AppConfig;

    constructor(configDirectory?: string) {
        this.configDir = configDirectory || getDefaultOSConfigDir();
        try {
            if (!fsSync.existsSync(this.configDir)) {
                fsSync.mkdirSync(this.configDir, { recursive: true });
            }
        } catch (e) {
            console.warn(`Could not create config directory ${this.configDir}:`, e);
        }
        this.configPath = path.join(this.configDir, "config.json");
        this.config = JSON.parse(JSON.stringify(defaultConfig));
    }

    async load(): Promise<AppConfig> {
        try {
            const data = await fs.readFile(this.configPath, "utf8");
            const loaded = JSON.parse(data);
            this.config = {
                ...defaultConfig,
                ...loaded,
                openai: { ...defaultConfig.openai, ...loaded.openai },
                claude: { ...defaultConfig.claude, ...loaded.claude },
                gemini: { ...defaultConfig.gemini, ...loaded.gemini },
                grok: { ...defaultConfig.grok, ...loaded.grok },
                customOpenAI: { ...defaultConfig.customOpenAI, ...loaded.customOpenAI },
                customClaude: { ...defaultConfig.customClaude, ...loaded.customClaude },
                customOpenAITargets: loaded.customOpenAITargets || defaultConfig.customOpenAITargets,
                customClaudeTargets: loaded.customClaudeTargets || defaultConfig.customClaudeTargets,
                lmStudioTarget: loaded.lmStudioTarget || defaultConfig.lmStudioTarget,
                lmStudioModelContextSizes: loaded.lmStudioModelContextSizes || defaultConfig.lmStudioModelContextSizes,
                customPricing: { ...defaultConfig.customPricing, ...loaded.customPricing },
                openProjects: loaded.openProjects || defaultConfig.openProjects
            };
        } catch {
            this.config = JSON.parse(JSON.stringify(defaultConfig));
            await this.save();
        }
        return this.config;
    }

    async save(): Promise<void> {
        try {
            if (!fsSync.existsSync(this.configDir)) {
                await fs.mkdir(this.configDir, { recursive: true });
            }
            await fs.writeFile(this.configPath, JSON.stringify(this.config, null, 2), "utf8");
        } catch (err) {
            console.error("Failed to save config:", err);
        }
    }

    getConfig(): AppConfig {
        return this.config;
    }

    async updateConfig(newConfig: Partial<AppConfig>): Promise<AppConfig> {
        this.config = {
            ...this.config,
            ...newConfig,
            openai: { ...this.config.openai, ...(newConfig.openai || {}) },
            claude: { ...this.config.claude, ...(newConfig.claude || {}) },
            gemini: { ...this.config.gemini, ...(newConfig.gemini || {}) },
            grok: { ...this.config.grok, ...(newConfig.grok || {}) },
            customOpenAI: { ...this.config.customOpenAI, ...(newConfig.customOpenAI || {}) },
            customClaude: { ...this.config.customClaude, ...(newConfig.customClaude || {}) },
            customOpenAITargets: newConfig.customOpenAITargets || this.config.customOpenAITargets,
            customClaudeTargets: newConfig.customClaudeTargets || this.config.customClaudeTargets,
            lmStudioTarget: newConfig.lmStudioTarget || this.config.lmStudioTarget,
            lmStudioModelContextSizes: newConfig.lmStudioModelContextSizes || this.config.lmStudioModelContextSizes,
            customPricing: { ...this.config.customPricing, ...(newConfig.customPricing || {}) },
            openProjects: newConfig.openProjects || this.config.openProjects
        };
        await this.save();
        return this.config;
    }
}
