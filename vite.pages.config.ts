import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsConfigPaths from "vite-tsconfig-paths";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/pushup_to_find/",
  plugins: [react(), tailwindcss(), tsConfigPaths()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
