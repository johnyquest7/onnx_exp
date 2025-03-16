const express = require('express');
const app = express();
const port = 3000; // Or any port you prefer

// Add headers for cross-origin isolation

app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  next();
});
app.use((req, res, next) => {
    if (req.url.endsWith('.wasm')) {
        res.set('Content-Type', 'application/wasm');
    }
    next();
});
// Serve static files from the current directory
app.use(express.static(__dirname));

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});