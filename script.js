
const modelPath = './model_q4.onnx';
const tokenizerPath = './tokenizer.json';
let session = null;
let tokenizer = null;

async function loadModelAndTokenizer() {
    try {
        // Check if ort is available.
        if (typeof ort === 'undefined') {
            throw new Error("ONNX Runtime Web (ort) is not loaded.  Make sure the <script> tag is correct and the library is loaded.");
        }

        // --- Model Loading ---
        console.log("Attempting to load model from:", modelPath); // Log the path
        session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        });
        console.log('Model loaded successfully.');

        // --- Tokenizer Loading ---
        console.log("Attempting to load tokenizer from:", tokenizerPath);
        const tokenizerResponse = await fetch(tokenizerPath);

        if (!tokenizerResponse.ok) {
            // More specific error message, including the URL
            throw new Error(`Failed to fetch tokenizer from ${tokenizerResponse.url}: ${tokenizerResponse.status} ${tokenizerResponse.statusText}`);
        }
        //Check Content type is correct.
        const contentType = tokenizerResponse.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") === -1) {
           throw new Error(`Fetched file from ${tokenizerPath} is wrong type: ${contentType}`);
        }

        const tokenizerData = await tokenizerResponse.json();
        tokenizer = new Tokenizer(tokenizerData);
        console.log('Tokenizer loaded successfully.');

        return { session, tokenizer };

    } catch (error) {
        console.error('Error loading model or tokenizer:', error);
        // Even more informative error message for the user:
        document.getElementById('output').textContent = `Error loading model or tokenizer: ${error.message}.  Check the developer console (F12) for more details.  
                                                      Make sure:
                                                      1.  '${modelPath}' and '${tokenizerPath}' are in the correct location.
                                                      2.  Your browser supports WebGPU or WASM.
                                                      3. The model files are not corrupted.`;
        document.getElementById('loading').style.display = 'none';
        document.getElementById('askButton').disabled = true; // Disable button
        throw error; // Re-throw for debugging purposes
    }
}


async function runInference(question) {
    if (!session || !tokenizer) {
        document.getElementById('output').textContent = 'Model or Tokenizer not loaded yet.';
        return;
    }

    try {
        // Preprocess
        const prompt = `<start_of_turn>user\n${question}<end_of_turn>\n<start_of_turn>model\n`;
        const tokens = tokenizer.encode(prompt);
        const inputIDs = new BigInt64Array(tokens.map(token => BigInt(token)));

        // Create input tensor
        const inputTensor = new ort.Tensor('int64', inputIDs, [1, inputIDs.length]);
        const attentionMask = new ort.Tensor('int64', new BigInt64Array(inputIDs.length).fill(1n), [1, inputIDs.length]);
        const positionIds = new ort.Tensor('int64', new BigInt64Array(inputIDs.length).map((_, i) => BigInt(i)), [1, inputIDs.length]);

        const feeds = {
            input_ids: inputTensor,
            attention_mask: attentionMask,
            position_ids: positionIds,
        };

        let outputText = '';
        const maxNewTokens = 150;
        let currentInputIDs = inputIDs;

        for (let i = 0; i < maxNewTokens; i++) {
            const currentInputTensor = new ort.Tensor('int64', currentInputIDs, [1, currentInputIDs.length]);
            const currentAttentionMask = new ort.Tensor('int64', new BigInt64Array(currentInputIDs.length).fill(1n), [1, currentInputIDs.length]);
            const currentPositionIds = new ort.Tensor('int64', new BigInt64Array(currentInputIDs.length).map((_, j) => BigInt(j)), [1, currentInputIDs.length]);

            const currentFeeds = {
                input_ids: currentInputTensor,
                attention_mask: currentAttentionMask,
                position_ids: currentPositionIds,
            };

            const results = await session.run(currentFeeds);
            const outputTensor = results.logits;

            if (!outputTensor || !outputTensor.data) {
                console.error("Output tensor is invalid:", outputTensor);
                throw new Error("Invalid output tensor received from the model.");
            }

            const sequenceLength = outputTensor.dims[1];
            const vocabSize = outputTensor.dims[2];
            const lastTokenLogits = outputTensor.data.slice((sequenceLength - 1) * vocabSize, sequenceLength * vocabSize);

            let nextTokenId = 0;
            let maxLogit = -Infinity;
            for (let j = 0; j < lastTokenLogits.length; j++) {
                if (lastTokenLogits[j] > maxLogit) {
                    maxLogit = lastTokenLogits[j];
                    nextTokenId = j;
                }
            }

            const decodedToken = tokenizer.decode([nextTokenId]);
            if (decodedToken === " <eos>") break;

            outputText += decodedToken;
            document.getElementById('output').textContent = outputText;

            const nextTokenIds = new BigInt64Array([BigInt(nextTokenId)]);
            currentInputIDs = new BigInt64Array([...currentInputIDs, ...nextTokenIds]);
        }

        return outputText;

    } catch (error) {
        console.error('Inference error:', error);
        document.getElementById('output').textContent = 'Error during inference. Check console for details.';
        throw error;
    }
}

class Tokenizer {
    constructor(tokenizerData) {
        this.tokenizerData = tokenizerData;
        this.vocab = tokenizerData.model.vocab;
        this.merges = tokenizerData.model.merges;

        this.idToToken = new Map();
        for (const token in this.vocab) {
            this.idToToken.set(this.vocab[token], token);
        }

        this.mergesMap = new Map();
        this.merges.forEach(merge => {
            const [part1, part2] = merge.split(" ");
            this.mergesMap.set(part1 + part2, merge.replace(" ", ""));
        });
    }

    encode(text) {
        const tokens = this.tokenize(text);
        let ids = tokens.map(token => {
            const normToken = this.normalizeToken(token);
            const id = this.vocab[normToken];
            return id !== undefined ? id : this.vocab["<unk>"];
        });
        ids = this.mergeTokens(ids);
        return ids;
    }

    decode(ids) {
        const tokens = ids.map(id => {
            let token = this.idToToken.get(id);
            if (!token) {
                return "<unk>";
            }
            token = token.replace(/\u2581/g, ' ');
            return token;
        });

        let text = tokens.join('');
        text = text.replace(/  /g, ' ');
        return text;
    }

    mergeTokens(ids) {
        while (true) {
            let minRank = Infinity;
            let minIndex = -1;

            for (let i = 0; i < ids.length - 1; i++) {
                const token1 = this.idToToken.get(ids[i]);
                const token2 = this.idToToken.get(ids[i + 1]);
                if (!token1 || !token2) continue;

                const combined = token1 + token2;
                const merged = this.mergesMap.get(combined);
                if (merged) {
                    const rank = this.merges.indexOf(token1 + " " + token2);
                    if (rank > -1 && rank < minRank) {
                        minRank = rank;
                        minIndex = i;
                    }
                }
            }

            if (minIndex === -1) break;
            const newToken = this.vocab[this.idToToken.get(ids[minIndex]) + this.idToToken.get(ids[minIndex + 1])];
            ids.splice(minIndex, 2, newToken);
        }
        return ids;
    }

    normalizeToken(token) {
        token = token.replace(/ /g, '\u2581');
        return token;
    }

    tokenize(text) {
        const basicTokens = text.split(/([\s.,!?])/).filter(s => s);
        let tokens = [];
        basicTokens.forEach(token => {
            if (token.match(/[\s.,!?]/)) {
                tokens.push(token);
            } else {
                tokens.push(...Array.from(token));
            }
        });
        return tokens;
    }
}



document.getElementById('askButton').addEventListener('click', async () => {
    const question = document.getElementById('input').value;
    if (!question.trim()) {
        alert('Please enter a question.');
        return;
    }

    document.getElementById('askButton').disabled = true;
    document.getElementById('loading').style.display = 'block';
    document.getElementById('output').textContent = '';

    try {
        if (!session) {
            await loadModelAndTokenizer();
        }
        const output = await runInference(question);

    } catch (error) {
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('askButton').disabled = false;
    }
});