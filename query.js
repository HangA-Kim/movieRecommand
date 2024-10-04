"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getUseBased = exports.getItemBased = exports.getMovies = void 0;
const child_process_1 = require("child_process");
const path_1 = __importDefault(require("path"));
const isDevelopment = () => {
    // console.log("NODE_ENV>", process.env.NODE_ENV);
    // npm install cross-env --save-dev
    // "dev": "cross-env NODE_ENV=development nodemon --exec ts-node ./index.ts"
    return process.env.NODE_ENV === "development";
};
// const pythonExePath = isDevelopment()
//   ? path.join(__dirname, ".conda", "python.exe")
//   : path.join(__dirname, ".conda", "python3");
const pythonExePath = isDevelopment()
    ? path_1.default.join(__dirname, ".conda", "python.exe")
    : path_1.default.join(__dirname, "venv", "bin", "python3");
// const pythonExePath = path.join(
//   '/home/ubuntu/miniconda',
//   'envs',
//   'myenv',
//   'bin',
//   'python3'
// );
function queryMovies(res, result) {
    let responseData = "";
    //파이썬 파일 수행 결과를 받아온다
    result.stdout.on("data", function (data) {
        responseData += data.toString();
        console.log(data.toString());
        // res.send(data.toString());
    });
    result.on("close", (code) => {
        if (code === 0) {
            // res.send(output);
            // res.status(200).json({ answer: output })
            console.log(responseData);
            const jsonResponse = JSON.parse(responseData);
            res.status(200).json(jsonResponse);
        }
        else {
            res.status(500).send({ error: `Child process exited with code ${code}` });
        }
    });
    result.stderr.on("data", (data) => {
        console.error(`stderr: ${data}`);
    });
}
/**************************************************************
 * export
 */
const getMovies = (res, ...args) => {
    const scriptPath = path_1.default.join(__dirname, "resolver.py");
    const result = (0, child_process_1.spawn)(pythonExePath, [scriptPath, ...args]);
    queryMovies(res, result);
};
exports.getMovies = getMovies;
const getItemBased = (res, ...args) => {
    const scriptPath = path_1.default.join(__dirname, "recommender.py");
    const result = (0, child_process_1.spawn)(pythonExePath, [scriptPath, ...args]);
    queryMovies(res, result);
};
exports.getItemBased = getItemBased;
const getUseBased = (res, ...args) => {
    const scriptPath = path_1.default.join(__dirname, "recommender.py");
    const result = (0, child_process_1.spawn)(pythonExePath, [scriptPath, ...args]);
    // 파이썬 스크립트로 JSON 데이터를 전달
    result.stdin.write(JSON.stringify(args[1]));
    result.stdin.end();
    queryMovies(res, result);
};
exports.getUseBased = getUseBased;
