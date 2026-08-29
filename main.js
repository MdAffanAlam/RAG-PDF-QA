import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import path from "node:path";
import { fileURLToPath } from "url";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class PdfQA {
  constructor({ model, pdfPath, chunkSize, chunkOverlap }) {
    this.model = model;
    this.pdfPath = pdfPath;
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
  }

  async initChatModel() {
    this.llm = new Ollama({
      model: this.model,
      temperature: 0,
    });

    // const response = await this.llm.invoke(
    //     "Why do parrots talk?"
    // );

    // console.log(response);

    await this.loadPdf();
    await this.split();

    this.embeddingModel = new OllamaEmbeddings({ model: "all-minilm:latest" });
    await this.createVectorStore();
    return this;
  }

  async loadPdf() {
    console.log("Loading Documents...");
    const loader = new PDFLoader(path.join(__dirname, this.pdfPath));

    this.docs = await loader.load();
    // console.log(this.docs);
    return this;
  }

  async split() {
    const splitter = new RecursiveCharacterTextSplitter({
      separators: " ",
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
    });
    this.texts = await splitter.splitDocuments(this.docs);
    // console.log(this.texts[3].pageContent);
  }

  async createVectorStore() {
    console.log("Creating document embeddings...");
    this.db = await MemoryVectorStore.fromDocuments(
      this.texts,
      this.embeddingModel
    );
    const retriever = this.db.asRetriever(3);

    const retrievedDocuments = await retriever.invoke("What is Font?");

    console.log("result  :-  ",retrievedDocuments);
  }
}

const pdfPath = "assets/CSS_Cheatsheet.pdf";
const obj = new PdfQA({
  model: "llama3.2:3b",
  pdfPath,
  chunkSize: 100,
  chunkOverlap: 0,
});
console.log(obj);
await obj.initChatModel();
