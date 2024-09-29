import React, {useEffect, useState, useRef} from 'react'
import { Bar, Line } from "react-chartjs-2";

export default function BarGraph({freq}) {
    const [data, setData] = useState(null);
    const [showChart, setShowChart] = useState(false);
    const initialized = useRef(false)

    async function fetchTrends(freq) {
        let res = await fetch(`http://localhost:5003/api/trendline?freq=${freq}`)
        let resJson = await res.json()
        
        let tData = {
            labels: Object.keys(resJson),
            datasets: [
                {
                    label: `${freq[0].toUpperCase() + freq.slice(1)} Consumption (MW)`,
                    data: Object.values(resJson).map(v => parseFloat(v)),
                    // backgroundColor: "#094C9D"
                    borderColor: "#094C9D",
                    tension: 0.2,
                    fill: false
                }
            ],
        }
        setData(tData)
        setShowChart(true)
    }


    const options = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                grid: { color: "rgba(0,0,0,0.1)" },
                ticks: {
                    color: "rgba(0,0,0,0.5)",
                }
            },
            y: {
                grid: { color: "rgba(0,0,0,0.1)" },
                ticks: { color: "rgba(0,0,0,0.5)" },
                beginAtZero: true,
            },
        },
        plugins: {
            legend: { display: true, position: "right", },
        },
        animations: false
    };


    useEffect(() => {
        
        if (!initialized.current) {
            initialized.current = true
            fetchTrends(freq)
        }

    }, [])

    return (showChart && <Line options={options} data={data} />)
}