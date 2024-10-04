import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import path from "path";
import { type Response } from "express";

const isDevelopment = (): boolean => {
  // console.log("NODE_ENV>", process.env.NODE_ENV);
  // npm install cross-env --save-dev
  // "dev": "cross-env NODE_ENV=development nodemon --exec ts-node ./index.ts"
  return process.env.NODE_ENV === "development";
};

// const pythonExePath = isDevelopment()
//   ? path.join(__dirname, ".conda", "python.exe")
//   : path.join(__dirname, ".conda", "python3");

const pythonExePath = isDevelopment()
  ? path.join(__dirname, ".conda", "python.exe")
  : path.join(__dirname, "venv", "bin", "python3");

// const pythonExePath = path.join(
//   '/home/ubuntu/miniconda',
//   'envs',
//   'myenv',
//   'bin',
//   'python3'
// );

function queryMovies(res: Response, result: ChildProcessWithoutNullStreams) {
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
    } else {
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

export const getMovies = (res: Response, ...args: string[]) => {
  const scriptPath = path.join(__dirname, "resolver.py");
  const result = spawn(pythonExePath, [scriptPath, ...args]);
  queryMovies(res, result);
};

export const getItemBased = (res: Response, ...args: string[]) => {
  const scriptPath = path.join(__dirname, "recommender.py");
  const result = spawn(pythonExePath, [scriptPath, ...args]);
  queryMovies(res, result);
};
