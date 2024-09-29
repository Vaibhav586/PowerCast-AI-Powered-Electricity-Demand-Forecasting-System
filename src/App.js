import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
} from "react-router-dom";
import Home from "./routes/Home";
import LogIn from "./routes/LogIn";
import Dashboard from "./routes/Dashboard";

export default function App() {
  return (
    <Router>
        <Routes>
          <Route path="/login" element={<LogIn />}/>
          <Route path="/dashboard" element={<Dashboard />}/>
          <Route path="/" element={<Home />}/>
        </Routes>
    </Router>
  );
}
