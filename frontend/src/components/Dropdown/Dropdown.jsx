import React from 'react'
import DropdownButton from '../DropdownButton/DropdownButton';
import DropdownContent from '../DropdownContent/DropdownContent';
import "./Dropdown.css";
import { useState, useEffect, useRef } from 'react';

const Dropdown = ({ buttonText, content }) => {
    
    const [open, setOpen] = useState(false);

    const dropdownRef = useRef();
    const buttonRef = useRef();
    const contentRef = useRef();

    const toggleDropdown = () => {
        setOpen((open) => !open);
    };

    useEffect(() => {
        const handler = (event) => {
            if(dropdownRef.current && 
            !dropdownRef.current.contains(event.target)){
                setOpen(false);
            }
        };

        document.addEventListener("click", handler);

        return () => {
            document.removeEventListener
            ("click", handler);
        };
    }, [])
    
    return (
        <div className="dropdown" ref={dropdownRef}>
            <DropdownButton ref={buttonRef} toggle={toggleDropdown} open={open}>
                {buttonText}
            </DropdownButton>
            <DropdownContent ref={contentRef} open={open}>
                {content}
            </DropdownContent>
        </div>
    );
};

export default Dropdown;