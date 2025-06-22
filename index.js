import express from 'express';
import dotenv from 'dotenv';
import { OpenAI } from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';

dotenv.config();

const app = express();
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = await pc.index(process.env.PINECONE_INDEX);
const namespace = "my-namespace";
const ns = index.namespace(namespace);

app.post('/embed', async (req, res) => {
  const { text, id } = req.body;

  try {
    const response = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
    });

    const vector = {
      id: id || crypto.randomUUID(),
      values: response.data[0].embedding,
      metadata: { text },
    };

    await ns.upsert([vector]);

    res.json({ message: 'Vector stored', vectorId: vector.id });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Upsert failed' });
  }
});

app.post('/query', async (req, res) => {
  const { query } = req.body;

  try {
    const response = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: query,
    });

    const queryVector = response.data[0].embedding;

    const result = await index.query({
      topK: 3,
      vector: queryVector,
      includeMetadata: true,
    });

    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Query failed' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`✅ Server running at http://localhost:${PORT}`));
