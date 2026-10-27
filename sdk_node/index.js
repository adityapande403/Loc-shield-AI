/**
 * LocalShield AI Node.js Client
 * Privacy-First Local PII Detection & Redaction Client
 */

class LocalShieldClient {
  /**
   * @param {Object} [options]
   * @param {string} [options.baseUrl='http://127.0.0.1:8000'] - LocalShield service URL
   * @param {string} [options.apiKey] - Optional API key (X-API-Key header)
   * @param {number} [options.timeoutMs=5000] - Request timeout in milliseconds
   */
  constructor(options = {}) {
    this.baseUrl = (options.baseUrl || "http://127.0.0.1:8000").replace(/\/+$/, "");
    this.apiKey = options.apiKey || null;
    this.timeoutMs = options.timeoutMs || 5000;
  }

  /**
   * Internal HTTP request helper with timeout and safe error formatting.
   * @private
   */
  async _request(endpoint, method = "GET", body = null) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };

    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errDetail = response.statusText;
        try {
          const errJson = await response.json();
          errDetail = errJson.detail || errJson.message || response.statusText;
        } catch (_) {}
        throw new Error(`LocalShield API Error (${response.status}): ${errDetail}`);
      }

      return await response.json();
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        throw new Error(`LocalShield request timed out after ${this.timeoutMs}ms`);
      }
      throw err;
    }
  }

  /**
   * Check liveness of the LocalShield service.
   * @returns {Promise<{ status: string, model_loaded: boolean, version: string }>}
   */
  async health() {
    return await this._request("/health");
  }

  /**
   * Check readiness of the LocalShield service.
   * @returns {Promise<{ ready: boolean, model_loaded: boolean, degraded_mode: boolean }>}
   */
  async ready() {
    return await this._request("/ready");
  }

  /**
   * Fetch versioned taxonomy of supported PII entities.
   * @returns {Promise<{ version: string, total_categories: number, categories: Object }>}
   */
  async getTaxonomy() {
    return await this._request("/taxonomy");
  }

  /**
   * Detect sensitive entities without modifying the original text.
   * @param {string} text - Input text to scan
   * @param {Object} [options]
   * @param {string[]} [options.categories] - Optional category filter
   * @param {number} [options.minConfidence=0.5] - Confidence threshold
   * @returns {Promise<{ entities: Array, total_detected: number, processing_time_ms: number }>}
   */
  async detect(text, options = {}) {
    if (typeof text !== "string") {
      throw new TypeError("text argument must be a string");
    }
    return await this._request("/detect", "POST", {
      text,
      categories: options.categories || null,
      min_confidence: options.minConfidence !== undefined ? options.minConfidence : 0.5,
    });
  }

  /**
   * Redact sensitive information from text.
   * @param {string} text - Input text to redact
   * @param {Object} [options]
   * @param {string[]} [options.categories] - Categories to redact
   * @param {'mask'|'pseudonymize'|'type_preserving'|'selective'} [options.mode='mask'] - Redaction mode
   * @param {boolean} [options.enableReversible=false] - Store encrypted reversible mapping
   * @param {string} [options.sessionId] - Session ID for reversible mapping
   * @returns {Promise<{ redacted_text: string, spans: Array, detected_count: number, redacted_count: number }>}
   */
  async redact(text, options = {}) {
    if (typeof text !== "string") {
      throw new TypeError("text argument must be a string");
    }
    return await this._request("/redact", "POST", {
      text,
      categories: options.categories || null,
      mode: options.mode || "mask",
      enable_reversible: Boolean(options.enableReversible),
      session_id: options.sessionId || null,
    });
  }

  /**
   * Redact a batch of texts in a single call.
   * @param {string[]} texts - Array of texts
   * @param {Object} [options]
   * @param {string[]} [options.categories]
   * @param {'mask'|'pseudonymize'|'type_preserving'|'selective'} [options.mode='mask']
   * @returns {Promise<{ results: Array, total_processed: number, total_processing_time_ms: number }>}
   */
  async redactBatch(texts, options = {}) {
    if (!Array.isArray(texts)) {
      throw new TypeError("texts argument must be an array of strings");
    }
    return await this._request("/redact/batch", "POST", {
      texts,
      categories: options.categories || null,
      mode: options.mode || "mask",
    });
  }

  /**
   * Analyze text for PII risk score and category breakdown.
   * @param {string} text
   * @returns {Promise<{ total_candidates: number, risk_score: number, category_breakdown: Object }>}
   */
  async analyze(text) {
    if (typeof text !== "string") {
      throw new TypeError("text argument must be a string");
    }
    return await this._request("/analyze", "POST", { text });
  }
}

module.exports = { LocalShieldClient };
