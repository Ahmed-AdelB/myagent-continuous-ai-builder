/* NOTE: Added minimal ESLint config for Vite React frontend (Codex). */
module.exports = {
  root: true,
  env: { browser: true, es2021: true, node: true },
  extends: [
    "eslint:recommended",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended"
  ],
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    ecmaFeatures: { jsx: true }
  },
  settings: { react: { version: "detect" } },
  plugins: ["react", "react-hooks"],
  rules: {
    "react/prop-types": "off"
  }
};
