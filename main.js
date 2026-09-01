import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import path from "node:path";
import { fileURLToPath } from "url";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
// import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { CloudClient } from "chromadb";
import "dotenv/config";

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
    console.log("---------------------------------------------------");
    console.log("---------------------------------------------------");

    const loader = new PDFLoader(path.join(__dirname, this.pdfPath));

    this.docs = await loader.load();
    // console.log(this.docs);
    return this;
  }

  async split() {
    console.log("Splitting Documents...");
    console.log("---------------------------------------------------");
    console.log("---------------------------------------------------");
    const splitter = new RecursiveCharacterTextSplitter({
      separators: " ",
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
    });
    this.texts = await splitter.splitDocuments(this.docs);
    // console.log(this.texts[2].pageContent);
  }

  async createVectorStore() {
    console.log("Creating document embeddings...");
    console.log("---------------------------------------------------");
    console.log("---------------------------------------------------");

    //this is for local database in langchain
    // this.db = await MemoryVectorStore.fromDocuments(
    //   this.texts,
    //   this.embeddingModel
    // );
    // console.log("db :-",this.db)
    // const retriever = this.db.asRetriever(2);

    // const retrievedDocuments = await retriever.invoke("What is Font?");

    // console.log("result  :-  ",retrievedDocuments);

    this.client = new CloudClient({
      apiKey: process.env.CHROMA_API_KEY,
      tenant: process.env.CHROMA_TENANT,
      database: process.env.CHROMA_DATABASE,
    });

    await this.client.heartbeat();

    console.log("Chroma Cloud connected!");
    console.log("---------------------------------------------------");
    console.log("---------------------------------------------------");

    await this.createCollection();
    return this;
  }

  async createCollection() {
    console.log("Create collection...");
    console.log("---------------------------------------------------");
    console.log("---------------------------------------------------");
    this.collection = await this.client.getOrCreateCollection({
      name: "pdf_documents",
      embeddingFunction: null,
    });
    console.log("Collection Name :- ", this.collection.name);
    console.log("---------------------------------------------------");
    console.log("---------------------------------------------------");
    await this.addDocuments();
    return this;
  }

  async addDocuments() {
    //Extract content from documents
    this.content = this.texts.map((doc) => doc.pageContent);

    //Unique ids
    this.ids = this.texts.map((_, index) => {
      return `ID${index}`;
    });

    //create embeddings for each documents
    this.embedds = await this.embeddingModel.embedDocuments(this.content);

    await this.collection.add({
      ids: this.ids,
      documents: this.content,
      embeddings: this.embedds,
    });

    console.log("PDF successfully stored in Chroma Cloud!");

    console.log("---------------------------------------------------");
    console.log("---------------------------------------------------");

    await this.queryDocuments();
    return this;
  }

  async queryDocuments() {
    //get documents from CHROMA DATABASE
    const queryTxt = "What is Font?";
    const queryEmbedding = await this.embeddingModel.embedQuery(queryTxt);
    // console.log("queryEmbedding :-",queryEmbedding)

    this.results = await this.collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: 4,
    });
    console.log("Results :- ", this.results);
    console.log("---------------------------------------------------");
    console.log("---------------------------------------------------");
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
