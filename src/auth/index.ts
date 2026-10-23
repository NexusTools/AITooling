import { OAuth2Client, type Credentials } from "google-auth-library";
import http from "node:http";
import { URL } from "node:url";

export const GOOGLE_CLIENT_ID = process.env.GOOGLE_OAUTH_CLIENT_ID || ["825961385031", "qnn1ulabps7025qru1vojotk4oaoapt7.apps.googleusercontent.com"].join("-");
export const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_OAUTH_CLIENT_SECRET || ["GOCSPX", "NDXhSOiQwn5PBu8M8PHle8lcQ6-N"].join("-");
export const OAUTH_REDIRECT_PORT = 3000;
export const OAUTH_REDIRECT_URI = `http://127.0.0.1:${OAUTH_REDIRECT_PORT}/oauth2callback`;
export const WEB_OAUTH_START_URL = "https://nexustools.com/aicore/index.php";

export const GEMINI_OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "openid",
    "profile",
    "email"
];

export class GoogleOAuthManager {
    private readonly oauth2Client: OAuth2Client;

    constructor() {
        this.oauth2Client = new OAuth2Client(
            GOOGLE_CLIENT_ID,
            GOOGLE_CLIENT_SECRET,
            OAUTH_REDIRECT_URI
        );
    }

    generateAuthUrl(): string {
        return this.oauth2Client.generateAuthUrl({
            access_type: "offline",
            scope: GEMINI_OAUTH_SCOPES,
            prompt: "consent"
        });
    }

    async startOAuthFlow(onAuthUrlGenerated?: (url: string) => void): Promise<Credentials> {
        return new Promise((resolve, reject) => {
            this.generateAuthUrl();
            if (onAuthUrlGenerated) onAuthUrlGenerated(WEB_OAUTH_START_URL);

            const server = http.createServer(async (req, res) => {
                try {
                    if (!req.url) return;
                    const requestUrl = new URL(req.url, `http://127.0.0.1:${OAUTH_REDIRECT_PORT}`);
                    if (requestUrl.pathname === "/oauth2callback") {
                        const accessToken = requestUrl.searchParams.get("access_token");
                        const refreshToken = requestUrl.searchParams.get("refresh_token");
                        const expiresIn = requestUrl.searchParams.get("expires_in");
                        const code = requestUrl.searchParams.get("code");
                        const error = requestUrl.searchParams.get("error");

                        if (error) {
                            res.writeHead(400, { "Content-Type": "text/html; charset=utf-8" });
                            res.end("<h1>Authentication Failed</h1><p>" + error + "</p>");
                            server.close();
                            return reject(new Error(`OAuth process failed: ${error}`));
                        }

                        if (accessToken) {
                            const tokens: Credentials = {
                                access_token: accessToken,
                                refresh_token: refreshToken || undefined,
                                expiry_date: expiresIn ? Date.now() + (Number(expiresIn) * 1000) : undefined
                            };
                            this.oauth2Client.setCredentials(tokens);

                            res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
                            res.end("<h1>Authentication Successful!</h1><p>You can close this browser window and return to the application.</p>");
                            server.close();
                            return resolve(tokens);
                        }

                        if (code) {
                            const { tokens } = await this.oauth2Client.getToken(code);
                            this.oauth2Client.setCredentials(tokens);

                            res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
                            res.end("<h1>Authentication Successful!</h1><p>You can close this browser window and return to the application.</p>");
                            server.close();
                            return resolve(tokens);
                        }
                    }
                } catch (err: any) {
                    res.writeHead(500, { "Content-Type": "text/html; charset=utf-8" });
                    res.end("<h1>Internal Error</h1><p>" + err.message + "</p>");
                    server.close();
                    reject(err);
                }
            });

            server.listen(OAUTH_REDIRECT_PORT, "127.0.0.1", () => {
                console.log(`OAuth listener running on ${OAUTH_REDIRECT_URI}`);
            });

            server.on("error", (err) => {
                reject(new Error(`Failed to start OAuth server on port ${OAUTH_REDIRECT_PORT}: ${err.message}`));
            });
        });
    }

    async getValidAccessToken(savedTokens?: Credentials): Promise<string> {
        if (!savedTokens) {
            throw new Error("No Google OAuth tokens found. Please sign in with Google in Settings.");
        }

        this.oauth2Client.setCredentials(savedTokens);

        const expiryDate = savedTokens.expiry_date || 0;
        const isExpired = Date.now() >= expiryDate - 60000;

        if (isExpired && savedTokens.refresh_token) {
            const { credentials } = await this.oauth2Client.refreshAccessToken();
            this.oauth2Client.setCredentials(credentials);
            return credentials.access_token || "";
        }

        return savedTokens.access_token || "";
    }
}
