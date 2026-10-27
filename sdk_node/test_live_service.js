/**
 * Live Node.js Client Integration Test against LocalShield FastAPI service
 */

const { LocalShieldClient } = require("./index.js");
const assert = require("assert");

async function main() {
  const client = new LocalShieldClient({ baseUrl: "http://127.0.0.1:8000", timeoutMs: 5000 });

  console.log("Testing Node.js client against live LocalShield service...");

  // 1. Health & Taxonomy
  const health = await client.health();
  assert.strictEqual(health.status, "healthy");
  console.log("-> /health: OK");

  const ready = await client.ready();
  assert.strictEqual(ready.ready, true);
  console.log("-> /ready: OK");

  const taxonomy = await client.getTaxonomy();
  assert(taxonomy.total_categories >= 20);
  console.log(`-> /taxonomy: OK (${taxonomy.total_categories} categories)`);

  // 2. Detection
  const det = await client.detect("Rahul Sharma called 9876543210 regarding account ABCPE1234F.");
  assert(det.total_detected >= 3);
  console.log(`-> /detect: OK (${det.total_detected} entities detected)`);

  // 3. Redaction (Single)
  const red = await client.redact("Rahul Sharma called 9876543210.", { mode: "mask" });
  assert(red.redacted_text.includes("[PHONE_NUMBER]"));
  assert(red.redacted_text.includes("[PERSON]"));
  console.log(`-> /redact: OK -> "${red.redacted_text}"`);

  // 4. Batch Redaction
  const batch = await client.redactBatch([
    "Priya Patel at 9876543210",
    "Meeting in Bengaluru office",
    "System memory OK",
  ]);
  assert.strictEqual(batch.total_processed, 3);
  console.log(`-> /redact/batch: OK (${batch.total_processed} items processed)`);

  // 5. Analyze
  const ana = await client.analyze("Secret AKIAIOSFODNN7EXAMPLE and phone 9876543210");
  assert(ana.risk_score > 0.8);
  console.log(`-> /analyze: OK (Risk Score: ${ana.risk_score})`);

  console.log("All Node.js live integration tests PASSED successfully!");
}

main().catch((err) => {
  console.error("Node.js live integration test failed:", err);
  process.exit(1);
});
