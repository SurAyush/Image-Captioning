import React from "react";

export default function Box({ heading, text }) {
  return (
    <div className="card">
        <h3>{heading}</h3>
        <p className="details">{text}</p>
    </div>
  );
}
