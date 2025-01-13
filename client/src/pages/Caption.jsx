import React, { useState } from "react";
import mouse from '../assets/mouse.jpg';
import Primary_Dropdown from "../components/Primary_Dropdown";
import Secondary_Dropdown from "../components/Secondary_Dropdown.jsx"

export default function Caption() {
    const [searchMethod, setSearchMethod] = useState('greedy');
    const [beamWidth, setBeamWidth] = useState('4');
    const [image, setImage] = useState(mouse);
    const [captions, setCaptions] = useState([]);

    const handleValueChange = (value) =>{
        setSearchMethod(value);
    }
    
    const handleBeamChange = (value) =>{
        setBeamWidth(value);
    }

    const handleGenerate = ()=>{
        console.log("ok");
    }

    const handleFileChange = (e)=>{
        const img_file = e.target.files[0];
    
        if (img_file.type.startsWith('image/')) {
            const objectUrl = URL.createObjectURL(e.target.files[0]);
            setImage(objectUrl);
        }
        else{
            alert("Please select an image file.");
        }
    }

    return (
        <div>
            <h1 className="heading">Caption with AI!</h1>
            <div className="options">
                <Primary_Dropdown onValueChange={handleValueChange}/>
                {searchMethod=='beam' && <Secondary_Dropdown onValueChange={handleBeamChange}/>}
                <div className="w-72">
                    <label htmlFor="file-input" className="block mb-1 text-gray-200">Choose Image File</label>
                    <input type="file" accept="image/*" id="file-input" onChange={handleFileChange} className="hidden"/>
                    <button onClick={() => document.getElementById('file-input').click()} className="w-full p-2 border rounded bg-transparent text-gray-200 border-gray-600 hover:border-gray-400 focus:outline-none cursor-pointer text-left"> 
                        {'Select a file...'}
                    </button>
                </div>
            </div>
            
            <div className="image-canvas"><img src={image}></img></div>
            
            <div className="w-96 genbtn">
                <button
                onClick={handleGenerate}
                className="w-full p-2 border rounded bg-transparent text-gray-200 border-gray-600 hover:border-gray-400 focus:outline-none cursor-pointer transition-colors duration-200"
                >
                Generate
                </button>
            </div>
            {captions.length != 0 && <div className="caption-canvas">
                <ul>
                    {captions.map((item, index) => (<li key={index}>{item}</li> ))}
                </ul>
            </div>}
        </div>
    );
}