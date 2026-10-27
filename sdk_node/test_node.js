/**
 * Node.js SDK Unit & Contract Tests
 */

const { LocalShieldClient } = require("./index.js");
const assert = require("assert");

async function runTests() {
  console.log("Starting Node.js SDK contract verification...");

  // 1. Client initialization
  const client = new LocalShieldClient({ baseUrl: "http://127.0.0.1:8000", timeoutMs: 3000 });
  assert.strictEqual(client.baseUrl, "http://127.0.0.1:8000");

  // 2. Input validation tests
  await assert.rejects(
    async () => await client.detect(null),
    /TypeError: text argument must be a string/,
    "Should reject non-string detect input"
  );

  await assert.rejects(
    async () => await client.redact(123),
    /TypeError: text argument must be a string/,
    "Should reject non-string redact input"
  );

  await assert.rejects(
    async () => await client.redactBatch("not an array"),
    /TypeError: texts argument must be an array of strings/,
    "Should reject non-array batch input"
  );

  console.log("Input validation tests: PASSED!");

  // 3. Timeout behavior test
  const timeoutClient = new LocalShieldClient({ baseUrl: "http://10.255.255.1", timeoutMs: 100 });
  await assert.rejects(
    async () => await timeoutClient.health(),
    /LocalShield request timed out|fetch failed/,
    "Should abort on timeout"
  );
  console.log("Timeout handling test: PASSED!");

  console.log("All Node.js SDK unit checks passed successfully!");
}

runTests().catch((err) => {
  console.error("Node.js test failure:", err);
  process.exit(1);
});
