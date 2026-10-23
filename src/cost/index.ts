import type { ModelPricing } from "../config/index.js";

export interface CostCalculation {
    model: string;
    inputTokens: number;
    outputTokens: number;
    inputCostUSD: number;
    outputCostUSD: number;
    totalCostUSD: number;
}

export const defaultPricingTable: Record<string, ModelPricing> = {
    // OpenAI Models
    "gpt-6-astra": { inputCostPer1M: 10.00, outputCostPer1M: 50.00 },
    "gpt-6.1-sol": { inputCostPer1M: 2.00, outputCostPer1M: 10.00 },
    "gpt-6-luna": { inputCostPer1M: 0.10, outputCostPer1M: 0.50 },
    "gpt-4o": { inputCostPer1M: 2.50, outputCostPer1M: 10.00 },
    "gpt-4o-mini": { inputCostPer1M: 0.15, outputCostPer1M: 0.60 },
    "o3-mini": { inputCostPer1M: 1.10, outputCostPer1M: 4.40 },
    "o1": { inputCostPer1M: 15.00, outputCostPer1M: 60.00 },
    "gpt-4-turbo": { inputCostPer1M: 10.00, outputCostPer1M: 30.00 },

    // Claude Models
    "claude-opus-4-8": { inputCostPer1M: 15.00, outputCostPer1M: 75.00 },
    "claude-sonnet-5": { inputCostPer1M: 3.00, outputCostPer1M: 15.00 },
    "claude-haiku-5-5": { inputCostPer1M: 0.10, outputCostPer1M: 0.50 },
    "claude-3-7-sonnet-latest": { inputCostPer1M: 3.00, outputCostPer1M: 15.00 },
    "claude-3-5-sonnet-20241022": { inputCostPer1M: 3.00, outputCostPer1M: 15.00 },

    // Gemini Models
    "gemini-3.8-flash": { inputCostPer1M: 0.75, outputCostPer1M: 3.75 },
    "gemini-3.7-flash": { inputCostPer1M: 0.75, outputCostPer1M: 3.75 },
    "gemini-3.6-flash": { inputCostPer1M: 0.75, outputCostPer1M: 3.75 },
    "gemini-3.5-flash-lite": { inputCostPer1M: 0.25, outputCostPer1M: 1.25 },
    "gemini-2.5-flash": { inputCostPer1M: 0.075, outputCostPer1M: 0.30 },
    "gemini-2.5-pro": { inputCostPer1M: 1.25, outputCostPer1M: 5.00 },

    // Grok Models
    "grok-4.7": { inputCostPer1M: 2.00, outputCostPer1M: 6.00 },
    "grok-2": { inputCostPer1M: 2.00, outputCostPer1M: 10.00 }
};

export class CostCalculator {
    static getModelPricing(model: string, customPricing?: Record<string, ModelPricing>): ModelPricing {
        const baseModel = model.split(":")[0];
        if (customPricing && customPricing[baseModel]) {
            return customPricing[baseModel];
        }
        if (defaultPricingTable[baseModel]) {
            return defaultPricingTable[baseModel];
        }
        return { inputCostPer1M: 0.00, outputCostPer1M: 0.00 };
    }

    static calculateCost(
        model: string,
        inputTokens: number,
        outputTokens: number,
        customPricing?: Record<string, ModelPricing>
    ): CostCalculation {
        const pricing = CostCalculator.getModelPricing(model, customPricing);
        const inputCostUSD = (inputTokens / 1_000_000) * pricing.inputCostPer1M;
        const outputCostUSD = (outputTokens / 1_000_000) * pricing.outputCostPer1M;
        const totalCostUSD = inputCostUSD + outputCostUSD;

        return {
            model,
            inputTokens,
            outputTokens,
            inputCostUSD,
            outputCostUSD,
            totalCostUSD
        };
    }

    static formatCostPromptContext(
        model: string,
        accumulatedTokens: { input: number; output: number },
        accumulatedCostUSD: number,
        customPricing?: Record<string, ModelPricing>
    ): string {
        const pricing = CostCalculator.getModelPricing(model, customPricing);
        return [
            `[SYSTEM COST CONTEXT]`,
            `Model: ${model}`,
            `Pricing: $${pricing.inputCostPer1M.toFixed(2)}/1M input tokens, $${pricing.outputCostPer1M.toFixed(2)}/1M output tokens`,
            `Session Consumption: ${accumulatedTokens.input} input tokens, ${accumulatedTokens.output} output tokens`,
            `Total Spent USD: $${accumulatedCostUSD.toFixed(6)}`
        ].join("\n");
    }
}
