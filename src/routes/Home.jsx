import { MDBContainer, MDBTypography, MDBIcon } from "mdb-react-ui-kit";
import React, { useEffect } from "react";
import TopMenu from "../components/TopMenu";
import Footer from "../components/Footer";

export default function Home() {
    useEffect(() => {
        document.cookie = "username=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
    }, [])

    return (
        <>
        <TopMenu />
        <div style={{
            backgroundImage: "url('/hero-bg.jpg')",
            backgroundPositionY: "75%",
            backgroundRepeat: "no-repeat",
            backgroundSize: "cover"
        }}>
            <MDBContainer style={{
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                color: "white",
                fontWeight: "bold",
                height: 480,
                padding: 12
            }}>
                <MDBTypography variant="h1" style={{fontSize: 64, marginBottom: 24}}>
                    Electricity Projection Tool
                </MDBTypography>
                <MDBTypography variant="h4" style={{marginBottom: 24}}>
                    Get accurate projections for electricity usage in Delhi for the current year.
                </MDBTypography>
            </MDBContainer>
        </div>

        <MDBContainer style={{
            display: "flex",
            flexDirection: "row",
            justifyContent: "space-between",
            padding: 12,
            paddingTop: 100,
            paddingBottom: 48,
        }}>
            <div style={{width: "50%", marginTop: 120}}>
                <MDBTypography variant="h1" style={{marginBottom: 12}}>
                    Introduction
                </MDBTypography>
                <MDBTypography variant="p"  style={{fontSize: 18}}>
                    As the capital city of India, Delhi experiences significant fluctuations in power demand throughout the year. These variations are influenced by multiple factors, including weather patterns, population growth, and economic activities. To ensure a stable and reliable power supply, our team has initiated a project to leverage historical data and advanced machine learning techniques to predict power usage across different times of the year and day.
                </MDBTypography>
            </div>
            <img src="https://akm-img-a-in.tosshub.com/indiatoday/images/story/202406/delhi-power-cut-112300488-16x9_0.jpeg?VersionId=jJ6n6cZ.WFt0WUg5ry_U.oNt8Umtejwd" alt="" style={{width: "50%",  marginRight: -120, height: 640, objectFit: "cover"}} />
        </MDBContainer>

        <MDBContainer style={{
            display: "flex",
            flexDirection: "row",
            justifyContent: "space-between",
            padding: 12,
            paddingTop: 48,
            paddingBottom: 160,
        }}>
            <img src="https://www.ibef.org/assets/images/Renewable-Energy-2.jpg" alt="" style={{width: "50%",  marginLeft: -100, marginTop: -120, height: 640, objectFit: "cover"}} />
            <div style={{width: "50%", marginTop: 40}}>
                <MDBTypography variant="h1" style={{marginBottom: 12}}>
                    Objectives
                </MDBTypography>
                <MDBTypography listUnStyled className='mb-0' style={{fontSize: 18}}>
                    <li className='mb-1'>
                        <MDBIcon icon='long-arrow-alt-right' className='me-2 text-info' />Accurate Power Demand Forecasting
                    </li>
                    <li className='mb-1'>
                        <MDBIcon icon='long-arrow-alt-right' className='me-2 text-info' />Resource Optimization
                    </li>
                    <li className='mb-1'>
                        <MDBIcon icon='long-arrow-alt-right' className='me-2 text-info' />Cost Reduction
                    </li>
                    <li className='mb-1'>
                        <MDBIcon icon='long-arrow-alt-right' className='me-2 text-info' />Infrastructure Planning
                    </li>
                    <li className='mb-1'>
                        <MDBIcon icon='long-arrow-alt-right' className='me-2 text-info' />Environmental Impact
                    </li>
                </MDBTypography>
            </div>
        </MDBContainer>

        <Footer />
        </>
    )
}