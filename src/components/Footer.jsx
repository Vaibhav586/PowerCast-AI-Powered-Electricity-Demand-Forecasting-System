import { MDBContainer, MDBTypography } from 'mdb-react-ui-kit'
import React from 'react'

export default function Footer() {
    return (
        <div style={{
            backgroundColor: "#000000",
            color: "white", 

        }}>
            <MDBContainer style={{
                padding: 12,
                marginRight: "auto",
                marginLeft: "auto",
                display: "flex",
                flexDirection: "row",
                justifyContent: "space-between"
            }}>
                <div style={{
                    marginTop: 24,
                    marginBottom: 24,
                }}>
                    <strong><MDBTypography variant='p'>Copyright © Team Flux 2024</MDBTypography></strong>
                    <MDBTypography variant='p' style={{
                        marginTop:0, marginBottom: 0
                    }}>Vaibhav</MDBTypography>
                    <MDBTypography variant='p' style={{
                        marginTop:0, marginBottom: 0
                    }}>Pushkin</MDBTypography>
                    <MDBTypography variant='p' style={{
                        marginTop:0, marginBottom: 0
                    }}>Sheeba</MDBTypography>
                    <MDBTypography variant='p' style={{
                        marginTop:0, marginBottom: 0
                    }}>Aditya Kulshrestha</MDBTypography>
                    <MDBTypography variant='p' style={{
                        marginTop:0, marginBottom: 0
                    }}>Garima</MDBTypography>
                </div>
                <div style={{
                    display: "flex",
                    flexDirection: "row"
                }}>
                    <img src="/sih-logo.png" style={{width: 200, objectFit: "contain"}} />
                    <img src="/srm-logo.png" style={{width: 90, objectFit: "contain", marginLeft: 48}} />

                </div>
            </MDBContainer>
        </div>
    )
}