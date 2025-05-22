// src/App.jsx
import React, { useEffect, useState, useRef } from "react";
import { DeckGL } from "deck.gl";
import { Map } from "react-map-gl/maplibre";
import { ScatterplotLayer, PathLayer } from "@deck.gl/layers";
import CompanyMarker from "./components/CompanyMarker";
import ProductList from "./components/ProductList";

// Usa un mapa gratuito de MapLibre
const MAP_STYLE = `https://api.maptiler.com/maps/streets/style.json?key=MZjUAQpw10B8E0nsKVQP`;

const INITIAL_VIEW_STATE = {
  latitude: 23.1136,
  longitude: -82.3666,
  zoom: 13,
  bearing: 0,
  pitch: 0,
};

function App() {
  const [vehicleData, setVehicleData] = useState({});
  const [connectionStatus, setConnectionStatus] = useState("Conectando...");
  const [errorMessage, setErrorMessage] = useState("");
  const [products, setProducts] = useState([]);
  const trailsRef = useRef({});
  const wsRef = useRef(null);

  // Definimos la empresa (almacén)
  const company = {
    id: "almacen_habana",
    name: "Almacén La Habana",
    lat: 23.1136,
    lon: -82.3666,
  };

  // Simulación de productos pedidos
  useEffect(() => {
    // Aquí podrías reemplazar por fetch a tu servidor
    const demoOrders = [
      { id: 1, name: "Camisetas", quantity: 20, express: false },
      { id: 2, name: "Zapatos", quantity: 10, express: true },
      { id: 3, name: "Gafas de sol", quantity: 5, express: false },
    ];
    setProducts(demoOrders);
  }, []);

  useEffect(() => {
    // Conexión WebSocket (igual que antes)
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket("ws://localhost:8765");
        wsRef.current = ws;

        ws.onopen = () => {
          setConnectionStatus("Conectado");
          setErrorMessage("");
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            const updated = {};
            data.vehicles.forEach((v) => {
              const prevTrail = trailsRef.current[v.id] || [];
              const newTrail = [...prevTrail.slice(-20), [v.lon, v.lat]];
              trailsRef.current[v.id] = newTrail;
              updated[v.id] = {
                id: v.id,
                position: [v.lon, v.lat],
                trail: newTrail,
                color: getVehicleColor(v.id),
              };
            });
            setVehicleData(updated);
          } catch (err) {
            setErrorMessage(`Error procesando datos: ${err.message}`);
          }
        };

        ws.onclose = () => {
          setConnectionStatus("Desconectado");
          setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = () => {
          setConnectionStatus("Error");
          setErrorMessage("Error en la conexión WebSocket");
        };
      } catch (err) {
        setConnectionStatus("Error");
        setErrorMessage(`Error de conexión: ${err.message}`);
        setTimeout(connectWebSocket, 3000);
      }
    };
    connectWebSocket();
    return () => wsRef.current?.close();
  }, []);

  const getVehicleColor = (id) => {
    const idNum = parseInt(id.replace("veh_", ""), 10);
    const palette = [
      [255, 0, 0], [0, 255, 0], [0, 0, 255],
      [255, 255, 0], [255, 0, 255], [0, 255, 255],
      [255, 165, 0], [128, 0, 128], [0, 128, 0], [139, 69, 19],
    ];
    return palette[idNum % palette.length];
  };

  const layers = [
    new ScatterplotLayer({
      id: "vehicles",
      data: Object.values(vehicleData),
      getPosition: (d) => d.position,
      getFillColor: (d) => d.color,
      getRadius: 50,
      radiusMinPixels: 5,
      radiusMaxPixels: 15,
    }),
    new PathLayer({
      id: "trails",
      data: Object.values(vehicleData),
      getPath: (d) => d.trail,
      getWidth: 3,
      getColor: (d) => d.color,
      widthMinPixels: 2,
      widthMaxPixels: 5,
    }),
  ];

  // Estilos para el panel de control
  const overlayStyle = {
    position: "absolute",
    top: 10,
    left: 10,
    zIndex: 100,
    padding: 12,
    backgroundColor: "rgba(255,255,255,0.8)",
    borderRadius: 4,
    maxWidth: 280,
    fontFamily: "Arial, sans-serif",
    fontSize: "0.9rem"
  };

  return (
    <>
      <div style={overlayStyle}>
        <div><strong>Empresa:</strong> {company.name}</div>
        <div><strong>Estado WS:</strong> {connectionStatus}</div>
        {errorMessage && <div style={{ color: "red" }}>{errorMessage}</div>}
        <div><strong>Vehículos:</strong> {Object.keys(vehicleData).length}</div>
        <ProductList products={products} />
      </div>

      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
      >
        <Map reuseMaps mapLib={import("maplibre-gl")} mapStyle={MAP_STYLE}>
          <CompanyMarker company={company} />
        </Map>
      </DeckGL>
    </>
  );
}

export default App;
