import express, { type Express } from "express";
import cors from "cors";
import { getMovies, getItemBased, getUseBased } from "./query";

// console.log(path.join(__dirname)); // 루트 경로. 배포때와 개발때의 경로가 달라서

const app: Express = express();
const port = 8080;
const allowedOrigins = ["http://localhost:3000", "http://localhost:3001"];
const options: cors.CorsOptions = {
  origin: allowedOrigins,
};
app.use(cors(options));
// app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.send("hello from node server");
});

app.get("/random/:count", (req, res) => {
  getMovies(res, "random", req.params.count);
});

app.get("/latest/:count", (req, res) => {
  getMovies(res, "latest", req.params.count);
});

app.get("/genres/:genre/:count", (req, res) => {
  getMovies(res, "genres", req.params.genre, req.params.count);
});

app.get("/item-based/:item", (req, res) => {
  getItemBased(res, "item-based", req.params.item);
});

app.post("/user-based", (req, res) => {
  getUseBased(res, "user-based", req.body);
});

app
  .listen(port, () => {
    console.log(`[server]: Server is running at <http://localhost>:${port}`);
  })
  .on("error", (error) => {
    throw new Error(error.message);
  });
