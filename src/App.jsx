// src/App.jsx
import React, { useEffect, useState, useRef } from "react";
import { DeckGL } from "deck.gl";
import { Map } from "react-map-gl/maplibre";
import { ScatterplotLayer, PathLayer } from "@deck.gl/layers";
import ProductList from "./components/ProductList";

// Mapa gratuito de MapLibre
const MAP_STYLE = `https://api.maptiler.com/maps/streets/style.json?key=MZjUAQpw10B8E0nsKVQP`;

// 1) Centrar el mapa en La Habana / Almacenes San José
const INITIAL_VIEW_STATE = {
  latitude: 23.1298784,    // Almacenes San José, Habana Vieja
  longitude: -82.3490351,
  zoom: 15,                // Zoom alto para ver detalle
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

  // 2) Definir la empresa en Almacenes San José
  const company = {
    id: "almacen_san_jose",
    name: "Almacenes San José",
    lat: 23.1298784,
    lon: -82.3490351,
  };

  // Simulación de pedidos (igual que antes)…
  useEffect(() => {
    const demoOrders = [
      { id: 1, name: "Camisetas", quantity: 20, express: false },
      { id: 2, name: "Zapatos", quantity: 10, express: true },
      { id: 3, name: "Gafas de sol", quantity: 5, express: false },
    ];
    setProducts(demoOrders);
  }, []);

  // Conexión WebSocket (idéntica a la versión previa)…
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket("ws://localhost:8765");
      wsRef.current = ws;
      ws.onopen = () => {
        setConnectionStatus("Conectado");
        setErrorMessage("");
      };
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          const updated = {};
          data.vehicles.forEach((v) => {
            const prev = trailsRef.current[v.id] || [];
            const trail = [...prev.slice(-20), [v.lon, v.lat]];
            trailsRef.current[v.id] = trail;
            updated[v.id] = {
              id: v.id,
              position: [v.lon, v.lat],
              trail,
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
    };
    connectWebSocket();
    return () => wsRef.current?.close();
  }, []);

  const getVehicleColor = (id) => {
    const n = parseInt(id.replace("veh_", ""), 10);
    const palette = [
      [255, 0, 0], [0, 255, 0], [0, 0, 255],
      [255, 255, 0], [255, 0, 255], [0, 255, 255],
      [255, 165, 0], [128, 0, 128], [0, 128, 0], [139, 69, 19],
    ];
    return palette[n % palette.length];
  };

  // Montar capas: empresa fija + vehículos + rutas
  const layers = [
    new ScatterplotLayer({
      id: "company-point",
      data: [company],
      getPosition: d => [d.lon, d.lat],
      getFillColor: [255, 87, 34],
      getRadius: 50,
      radiusMinPixels: 10,
      pickable: false,
    }),
    new ScatterplotLayer({
      id: "vehicles",
      data: Object.values(vehicleData),
      getPosition: d => d.position,
      getFillColor: d => d.color,
      getRadius: 50,
      radiusMinPixels: 5,
      radiusMaxPixels: 15,
    }),
    new PathLayer({
      id: "trails",
      data: Object.values(vehicleData),
      getPath: d => d.trail,
      getColor: d => d.color,
      getWidth: 3,
      widthMinPixels: 2,
      widthMaxPixels: 5,
    }),
  ];

  const overlayStyle = {
    position: "absolute",
    top: 10, left: 10, zIndex: 100,
    padding: 12, backgroundColor: "rgba(255,255,255,0.8)",
    borderRadius: 4, maxWidth: 280,
    fontFamily: "Arial, sans-serif", fontSize: "0.9rem"
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
        <Map reuseMaps mapLib={import("maplibre-gl")} mapStyle={MAP_STYLE} />
      </DeckGL>
    </>
  );
}

export default App;
