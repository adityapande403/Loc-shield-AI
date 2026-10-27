/**
 * Example: Node.js client integration with LocalShield AI before sending text to external LLM
 */

const { LocalShieldClient } = require("../sdk_node/index.js");

async function run() {
  const client = new LocalShieldClient({ baseUrl: "http://127.0.0.1:8000" });

  const rawUserQuery =
    "Hi, I am Rahul Sharma. My phone is 9876543210 and my PAN is ABCPE1234F. Can you review my account?";

  console.log("Original Query from User:\n", rawUserQuery);

  // Redact before sending to any external API
  const redactResult = await client.redact(rawUserQuery, {
    mode: "mask",
  });

  console.log("\nRedacted Text (Safe for External LLM):\n", redactResult.redacted_text);
  console.log(`\nDetected ${redactResult.detected_count} sensitive entities locally in ${redactResult.processing_time_ms}ms.`);
}

run().catch(console.error);
