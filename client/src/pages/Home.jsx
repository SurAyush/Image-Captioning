import React from "react";
import Box from '../components/Box';
import {Link} from 'react-router-dom';

export default function Home() {
    return (
        <div>
        <h1 className="heading">Image Captioning App</h1>
        <div className="lineofbox">
            <Link to="https://chatgpt.com/" className="link">
                <Box heading="Blog" text="Read our Medium blog where we discuss about the architecture of the model in details!!!" />
            </Link>
            <Link to="/caption" className="link">
                <Box heading="Image Captioner" text="Try out our image captioning model with images of your choice!" />
            </Link>
            <Link to="https://github.com/SurAyush/Image-Captioning" className="link">
                <Box heading="GitHub Repository" text="Check out the repository of the project with all the codes of training, inferencing, server and much more!!!" />
            </Link>
        </div>
        </div>
    );
}