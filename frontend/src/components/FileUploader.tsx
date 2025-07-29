import { ChangeEvent, useState } from "react";
import axios from 'axios';
import Dropdown from "./Dropdown/Dropdown";
import DropdownItem from "./DropdownItem/DropdownItem";

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

function FileUploader() {

    const [file, setFile] = useState<File | null>(null);
    const [status, setStatus] = useState<UploadStatus>('idle');
    const [uploadProgress, setUploadProgress] = useState(0);

    function handleFileChange(e: ChangeEvent<HTMLInputElement>){
        if (e.target.files){
            setFile(e.target.files[0]);
        }
    }

    async function handleFileUpload() {
        if(!file){
            return;
        }

        setStatus("uploading");
        setUploadProgress(0);

        const formData = new FormData();
        formData.append("file", file);

        try {
            await axios.post("https://httpbin.org/post", formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    const progress = progressEvent.total ? 
                    Math.round((progressEvent.loaded * 100 / progressEvent.total))
                    : 0;
                    setUploadProgress(progress);
                }
            });

            setStatus("success");
            setUploadProgress(100);
        } catch {
            setStatus('error');
            setUploadProgress(0);
        }
    }

    const notes = 
    ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"];

    const keys = ["Major", "Minor"];

    return (
        <div className="flex">
            <div className="flex flex-col justify-center items-center mt-10 space-y-4">
                <input className="bg-white px-30 py-12 rounded-2xl shadow-lg"
                type="file" onChange={handleFileChange}/>
                {file && (
                    <div className="mb-4 text-sm space-y-2 text-center">
                        <p>File name: {file.name}</p>
                        <p>Size: {(file.size / 1024).toFixed(2)}</p>
                        <p>Type: {file.type}</p>
                    </div>
                )}

                {file && status !== "uploading" && 
                    <button className="font-light bg-white shadow-md mt-5 p-4 rounded-3xl" 
                    onClick={handleFileUpload}>Upload</button>
                }

                {status === 'uploading' && (
                    <div className="space-y-2">
                        <div className="h-2.5 w-full rounded-full bg-gray-200">
                            <div 
                                className="h-2.5 rounded-full bg-black transition-all duration-300"
                                style={{ width: `${uploadProgress}%` }}>
                            </div>
                        </div>
                        <p className="text-sm text-black">{uploadProgress}% uploaded</p>
                    </div>
                )}

                {status === 'success' && (
                    <p className="mt-2 text-sm text-center">
                        File uploaded successfully.
                    </p>
                )}

                {status === 'error' && (
                    <p className="mt-2 text-sm text-center">
                        Upload failed. Please try again.
                    </p>
                )}
            </div>
            <div className="flex mt-15 px-4 space-x-1">
                <Dropdown 
                buttonText="C" 
                content={<>
                {
                    notes.map(note => <DropdownItem
                    key={note}>
                        {`${note}`}
                    </DropdownItem>)
                }
                </>}
                />
                <Dropdown 
                buttonText="Major" 
                content={<>{
                    keys.map(key => <DropdownItem
                    key={key}>
                        {`${key}`}
                    </DropdownItem>)
                }</>}
                />
            </div>
        </div>
        
    )
}

export default FileUploader;