/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        serif: ["'Crimson Pro'", "Georgia", "Times New Roman", "serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "'Fira Code'", "monospace"],
      },
      colors: {
        brand: {
          DEFAULT: "#1f4e79",
          light: "#5a86b3",
          dark: "#102f4a",
        },
        accent: {
          gold: "#e6a730",
          rose: "#c44e52",
          green: "#3a8c5d",
          violet: "#9c7cb5",
        },
        ink: "#1a1a1a",
        paper: "#fafaf7",
      },
    },
  },
  plugins: [],
};
