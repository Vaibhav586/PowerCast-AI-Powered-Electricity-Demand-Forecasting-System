import React, { useRef } from 'react'
import { MDBTypography, MDBInputGroup, MDBBtn } from 'mdb-react-ui-kit'
import { Link } from 'react-router-dom'

function setCookie(cookieName, cookieValue) {
    const year = 60 * 60 * 24 * 365;
    document.cookie = `${cookieName}=${cookieValue}; max-age=${year}; path=/; SameSite=Lax; Secure;`;

    const cookie = document.cookie
      .split("; ")
      .find((row) => row.startsWith(cookieName))
      .split("=")[1];
    return cookie;
  }

export default function Login() {
    const userRef = useRef(null)

    function handleSubmit(e) {
        e.preventDefault()
        setCookie("username", userRef.current.value)
        e.target.submit()
    }

    return (
        <div style={{
            backgroundColor: "#162B49",
            color: "white",
            height: "100vh",
            padding: 48
        }}>
            <div style={{
                backgroundColor: "white",
                color: "black",
                width: 550,
                borderRadius: 12,
                marginRight: "auto",
                marginLeft: "auto",
                padding: 32,
                paddingTop: 20,
            }}>
                <div style={{
                    marginRight: "auto",
                    marginLeft: "auto",
                    display: 'flex',
                    flexDirection: "column",

                }}>
                    <Link to="/">
                        <img src="hero-1.png" alt="" style={{
                            objectFit: "contain",
                            width: "100%",
                            height: 100,
                            marginBottom: 12
                        }} />
                    </Link>
                    <MDBTypography style={{fontWeight: "bold", fontSize: 14, textAlign: "center"}} variant='h5'>Government of National Capital Territory of Delhi</MDBTypography>
                    <MDBTypography style={{fontWeight: "bold", fontSize: 14, textAlign: "center"}} variant='h5'>राष्ट्रीय राजधानी क्षेत्र दिल्ली सरकार</MDBTypography>
                </div>

                <form  onSubmit={handleSubmit} action="/dashboard" className='mt-3'>
                    <label htmlFor='username' className='form-label'>
                        Username <span style={{color: "red"}}>*</span>
                    </label>
                    <MDBInputGroup className='mb-3'>
                        <input className='form-control' id='username' type='text' ref={userRef}/>
                    </MDBInputGroup>
                    <label htmlFor='password' className='form-label'>
                        Password <span style={{color: "red"}}>*</span>
                    </label>
                    <MDBInputGroup className='mb-3'>
                        <input className='form-control' id='password' type='password' />
                    </MDBInputGroup>
                    <MDBBtn style={{width: "100%", marginTop: 12}}>Login</MDBBtn>
                </form>
            </div>

        </div>
    )
}