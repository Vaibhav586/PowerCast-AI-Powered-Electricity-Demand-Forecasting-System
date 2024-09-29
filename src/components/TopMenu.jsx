import { MDBContainer, MDBTypography } from 'mdb-react-ui-kit'
import React from 'react'
import { Link } from 'react-router-dom'

export default function TopMenu({user}) {
    return (
        <nav style={{
            backgroundColor: "#fff",
        }}>
            <div style={{
                backgroundColor: "#00245B",
                color: "white",
            }}>
                <MDBContainer style={{
                    display: "flex",
                    flexDirection: "row",
                    justifyContent: "flex-end",
                    paddingRight: 12,
                    paddingLeft: 12,
                }}>
                    <Link to={user ? "/" : "/login"}>
                        {user && <span style={{color: "white", marginRight: 12}}>{user}</span>}
                        <button style={{
                            backgroundColor: "#FFC303",
                            color: "black",
                            paddingRight: 16, paddingLeft: 16,
                            paddingTop: 8, paddingBottom: 8,
                            border: 0,
                            fontSize: 14,
                            fontWeight: "bold"
                        }}>
                            {user ? "Logout" : "Login"}
                        </button>
                    </Link>
                </MDBContainer>
            </div>
            <MDBContainer style={{
                display: "flex",
                flexDirection: "row",
                padding: 12,
            }}>
                <Link to="/">
                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center"}}>
                        <img src="/hero-1.png" alt="" />
                        <div style={{marginLeft: 24, color: "black", fontWeight: "bold"}}>
                            <MDBTypography style={{fontWeight: "bold"}} variant='h5'>Government of National Capital Territory of Delhi</MDBTypography>
                            <MDBTypography style={{fontWeight: "bold"}} variant='h5'>राष्ट्रीय राजधानी क्षेत्र दिल्ली सरकार</MDBTypography>
                        </div>
                    </div>
                </Link>
                <img src="/hero-2.png" alt="" style={{marginLeft: "auto", objectFit: "contain", width: "180px", marginRight: 16 }} />
                <img src="/sih-logo.png" alt="" style={{objectFit: "contain", width: "160px" }} />

            </MDBContainer>
        </nav>
    )
}