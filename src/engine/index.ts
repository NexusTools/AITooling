import type { BaseAIProvider, ChatHistoryItem } from "../providers/index.js";

export class EngineQueue {
    private busy = false;
    private provider: BaseAIProvider;
    private readonly queue: ((history: ChatHistoryItem[], prompt: string) => Promise<string>)[] = [];

    constructor(provider: BaseAIProvider) {
        this.provider = provider;
    }

    setProvider(provider: BaseAIProvider): void {
        this.provider = provider;
    }

    async prompt(history: ChatHistoryItem[], promptText: string): Promise<string> {
        if (this.busy) {
            return new Promise((resolve, reject) => {
                this.queue.push(async (h, p) => {
                    try {
                        const response = await this.provider.prompt(h, p);
                        resolve(response.text);
                        return response.text;
                    } catch (err) {
                        reject(err);
                        throw err;
                    }
                });
            });
        }
        this.busy = true;
        try {
            const response = await this.provider.prompt(history, promptText);
            return response.text;
        } finally {
            this.busy = false;
            const nextWork = this.queue.shift();
            if (nextWork) {
                nextWork(history, promptText);
            }
        }
    }
}
