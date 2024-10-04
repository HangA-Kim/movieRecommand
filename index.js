"use strict";
var __importDefault =
  (this && this.__importDefault) ||
  function (mod) {
    return mod && mod.__esModule ? mod : { default: mod };
  };
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const query_1 = require("./query");
// console.log(path.join(__dirname)); // 루트 경로. 배포때와 개발때의 경로가 달라서
const app = (0, express_1.default)();
const port = 8080;
const allowedOrigins = ["http://localhost:3000", "http://localhost:3001"];
const options = {
  origin: allowedOrigins,
};
app.use((0, cors_1.default)(options));
// app.use(cors());
app.use(express_1.default.json());
app.get("/", (req, res) => {
  res.send("hello from node server");
});
app.get("/random/:count", (req, res) => {
  (0, query_1.getMovies)(res, "random", req.params.count);
});
app.get("/latest/:count", (req, res) => {
  (0, query_1.getMovies)(res, "latest", req.params.count);
});
app.get("/genres/:genre/:count", (req, res) => {
  (0, query_1.getMovies)(res, "genres", req.params.genre, req.params.count);
});
app.get("/item-based/:item", (req, res) => {
  (0, query_1.getItemBased)(res, "item-based", req.params.item);
});
app.post("/user-based", (req, res) => {
  (0, query_1.getUseBased)(res, "user-based", req.body);
});
app
  .listen(port, "0.0.0.0", () => {
    console.log(`[server]: Server is running at <http://0.0.0.0>:${port}`);
  })
  .on("error", (error) => {
    throw new Error(error.message);
  });
