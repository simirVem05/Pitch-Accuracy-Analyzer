import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import FileUploader from './components/FileUploader'

const App = () => {
  return (
    <div className="w-screen h-screen bg-gray-100 flex flex-col items-center justify-center">
      <h1 className="mt-4 absolute top-5 left-5 text-3xl font-light">
        Pitch Accuracy Analyzer
      </h1>
      <h2 className="font-light text-xl">
        Welcome!
      </h2>
      <h3 className="text-lg font-light mt-4">
        Upload a recording of your vocals for pitch analysis and
      </h3>
      <h4 className="font-light mt-1 text-lg">select the key of your vocals.</h4>
      <FileUploader/>
    </div>

  )
}

export default App;
