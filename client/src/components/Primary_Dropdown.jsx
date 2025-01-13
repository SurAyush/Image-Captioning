import React, { useState } from 'react';

export default function Primary_Dropdown ({onValueChange}) {
  const [searchMethod, setSearchMethod] = useState('greedy');

  const handleChange = (e)=>{
    setSearchMethod(e.target.value);
    onValueChange(e.target.value);   
  }

  return (
    <div className="w-72">
      <label htmlFor="search-method" className="block mb-1 text-gray-200">
        Choose Searching Technique
      </label>
      <select
        id="search-method"
        value={searchMethod}
        onChange={handleChange}
        className="w-full p-2 border rounded bg-transparent text-gray-200 border-gray-600 focus:border-gray-400 focus:outline-none appearance-none cursor-pointer"
      >
        <option value="greedy" className="bg-gray-800">Greedy Search</option>
        <option value="beam" className="bg-gray-800">Beam Search</option>
      </select>
    </div>
  );
};
