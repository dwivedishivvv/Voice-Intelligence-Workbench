import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Outlet } from "react-router-dom";
import Layout from "@/components/layout";
import Landing from "./pages/Landing";
import Library from "./pages/Library";
import Ask from "./pages/Ask";
import Upload from "./pages/Upload";
import Live from "./pages/Live";
import Graph from "./pages/Graph";
import ClipDetail from "./pages/ClipDetail";
import Speakers from "./pages/Speakers";
import Settings from "./pages/Settings";
import "./style.css";

const Shell = () => (
  <Layout>
    <Outlet />
  </Layout>
);

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* "/" is the marketing landing page — full-bleed, outside the app shell. */}
        <Route path="/" element={<Landing />} />
        <Route element={<Shell />}>
          <Route path="/library" element={<Library />} />
          <Route path="/review" element={<Library />} />
          <Route path="/ask" element={<Ask />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/live" element={<Live />} />
          <Route path="/graph" element={<Graph />} />
          <Route path="/clips/:id" element={<ClipDetail />} />
          <Route path="/speakers" element={<Speakers />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
