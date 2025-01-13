import React, { useState } from 'react';

export default function Secondary_Dropdown({onValueChange}) {
  const [beamWidth, setBeamWidth] = useState('4');

  const handleChange = (e)=>{
    setBeamWidth(e.target.value);
    onValueChange(e.target.value);   
  }

  return (
    <div className="w-48">
      <label htmlFor="beam-size" className="block mb-1 text-gray-200">
        Choose Beam Width
      </label>
      <select
        id="beam-width"
        value={beamWidth}
        onChange={handleChange}
        className="w-full p-2 border rounded bg-transparent text-gray-200 border-gray-600 focus:border-gray-400 focus:outline-none appearance-none cursor-pointer"
      >
        <option value='2' className="bg-gray-800">2</option>
        <option value='3' className="bg-gray-800">3</option>
        <option value='4' className="bg-gray-800">4</option>
        <option value='5' className="bg-gray-800">5</option>
        <option value='6' className="bg-gray-800">6</option>
        <option value='7' className="bg-gray-800">7</option>
        <option value='8' className="bg-gray-800">8</option>
        <option value='9' className="bg-gray-800">9</option>
      </select>
    </div>
  );
};

