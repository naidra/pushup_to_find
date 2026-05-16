import { createFileRoute } from "@tanstack/react-router";
import { VisionApp } from "@/components/VisionApp";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "Push-up Counter" },
      {
        name: "description",
        content:
          "Real-time push-up counting from pose tracking, running 100% in your browser via MediaPipe WASM.",
      },
    ],
  }),
});

function Index() {
  return <VisionApp />;
}
