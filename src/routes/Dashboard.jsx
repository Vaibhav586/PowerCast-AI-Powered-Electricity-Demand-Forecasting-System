import React, { useState, useEffect, useRef } from "react";
import { MDBContainer, MDBTypography } from "mdb-react-ui-kit";
import TopMenu from "../components/TopMenu";
import { Chart } from 'chart.js/auto';
import { useNavigate } from 'react-router-dom';
import Footer from "../components/Footer";
import {
  MDBIcons,
  MDBTabs,
  MDBTabsItem,
  MDBTabsLink,
  MDBTabsContent,
  MDBTabsPane
} from 'mdb-react-ui-kit';
import BarGraph from "../components/BarGraph";
import { chart } from "react-chartjs-2"

Chart.register();

export default function Dashboard() {
    const [user, setUser] = useState("");
    const navigate = useNavigate();

    useEffect(() => {
        const u = document.cookie
        if (u == "") { console.log("empty"); navigate("/login", { replace: true }) }
        else { setUser(u
            .split('; ')
            .find((row) => row.startsWith('username='))
            .split('=')[1]) }

    }, [])

    const [iconsActive, setIconsActive] = useState('tab1');

    const handleIconsClick = (value) => {
        if (value === iconsActive) {
        return;
        }

        setIconsActive(value);
    };
    
    return (
        <>
        <TopMenu user={user} />
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
                height: 240,
            }}>
                <MDBTypography variant="h1" style={{ fontSize: 64, marginBottom: 24 }}>Dashboard</MDBTypography>
            </MDBContainer>
        </div>
        
        <MDBContainer style={{
            padding: 12,
            marginTop: 48,
            marginBottom: 72,
        }}>
            <MDBTypography variant="h2" style={{marginBottom: 12}}>Projected Consumption</MDBTypography>
            <MDBTabs className='mb-3'>
                <MDBTabsItem>
                    <MDBTabsLink onClick={() => handleIconsClick('tab1')} active={iconsActive === 'tab1'}>
                        30 Days
                    </MDBTabsLink>
                </MDBTabsItem>
                <MDBTabsItem>
                    <MDBTabsLink onClick={() => handleIconsClick('tab2')} active={iconsActive === 'tab2'}>
                        8 Weeks
                    </MDBTabsLink>
                </MDBTabsItem>
                <MDBTabsItem>
                    <MDBTabsLink onClick={() => handleIconsClick('tab3')} active={iconsActive === 'tab3'}>
                        12 Months
                    </MDBTabsLink>
                </MDBTabsItem>
            </MDBTabs>

            <MDBTabsContent>
                <MDBTabsPane open={iconsActive === 'tab1'} style={{height: 640, width: "100%"}}>
                    <BarGraph freq="daily" />
                </MDBTabsPane>
                <MDBTabsPane open={iconsActive === 'tab2'} style={{height: 640, width: "100%"}}>
                    <BarGraph freq="weekly" />
                </MDBTabsPane>
                <MDBTabsPane open={iconsActive === 'tab3'} style={{height: 640, width: "100%"}}>
                    <BarGraph freq="monthly" />
                </MDBTabsPane>
            </MDBTabsContent>
        </MDBContainer>
        <Footer />
        </>
    )
}