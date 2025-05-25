// src/App.jsx
import React, { useEffect, useState, useRef } from "react";
import { DeckGL } from "deck.gl";
import { Map } from "react-map-gl/maplibre";
import { ScatterplotLayer, PathLayer } from "@deck.gl/layers";
import ProductList from "./components/ProductList";
import TruckList from "./components/TruckList";
import { INITIAL_TRUCKS } from "./data/trucks";
import { assignOrdersToTrucks } from "./utils/assignOrders";

// Mapa MapLibre
const MAP_STYLE = `https://api.maptiler.com/maps/streets/style.json?key=MZjUAQpw10B8E0nsKVQP`;

// Vista inicial: Almacenes San José, La Habana
const INITIAL_VIEW_STATE = {
  latitude: 23.1298784,
  longitude: -82.3490351,
  zoom: 15,
  bearing: 0,
  pitch: 0,
};

function App() {
  const [products, setProducts] = useState([]);
  const [trucks, setTrucks] = useState([]);
  const [vehicleData, setVehicleData] = useState({});
  const trailsRef = useRef({});
  const wsRef = useRef(null);

  // 1) Cargamos pedidos de demo
  useEffect(() => {
    const demo = [
      { id: 1, name: "Camisetas", quantity: 20, express: false },
      { id: 2, name: "Zapatos", quantity: 10, express: true },
      { id: 3, name: "Gafas de sol", quantity: 5, express: false },
      { id: 4, name: "Sombreros", quantity: 30, express: false },
      { id: 5, name: "Bolsos", quantity: 25, express: true },
      { id: 6, name: "Bufandas", quantity: 15, express: false }
    ];
    setProducts(demo);
  }, []);

  // 2) Asignamos pedidos a camiones, inicializamos posición/color
  useEffect(() => {
    if (products.length === 0) return;
    // inicial
    const colored = INITIAL_TRUCKS.map((t, i) => ({
      ...t,
      position: [-82.3490351, 23.1298784],      // parten del almacén
      color: [(i + 1) * 40 % 255, (i + 2) * 70 % 255, (i + 3) * 100 % 255]
    }));
    const assigned = assignOrdersToTrucks(products, colored);
    setTrucks(assigned);
  }, [products]);

  // 3) Conexión WebSocket (igual que antes) para vehículos
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket("ws://localhost:8765");
      wsRef.current = ws;
      ws.onmessage = e => {
        const { vehicles } = JSON.parse(e.data);
        const updated = {};
        vehicles.forEach(v => {
          const prev = trailsRef.current[v.id] || [];
          const trail = [...prev.slice(-20), [v.lon, v.lat]];
          trailsRef.current[v.id] = trail;
          updated[v.id] = {
            position: [v.lon, v.lat],
            trail,
            color: getVehicleColor(v.id)
          };
        });
        setVehicleData(updated);
      };
      ws.onclose = () => setTimeout(connect, 3000);
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  const getVehicleColor = id => {
    const n = parseInt(id.replace("veh_", ""), 10);
    const pal = [
      [255, 0, 0], [0, 255, 0], [0, 0, 255],
      [255, 255, 0], [255, 0, 255], [0, 255, 255]
    ];
    return pal[n % pal.length];
  };

  // 4) Capas DeckGL: empresa, camiones, vehículos, rutas
  const layers = [
    // almacén
    new ScatterplotLayer({
      id: "company",
      data: [1],
      getPosition: () => [-82.3490351, 23.1298784],
      getFillColor: [255, 87, 34],
      getRadius: 25,
      radiusMinPixels: 10
    }),
    // camiones
    new ScatterplotLayer({
      id: "trucks",
      data: trucks,
      getPosition: t => t.position,
      getFillColor: t => t.color,
      getRadius: 100,
      radiusMinPixels: 8
    }),
    // vehículos externos
    new ScatterplotLayer({
      id: "vehicles",
      data: Object.values(vehicleData),
      getPosition: d => d.position,
      getFillColor: d => d.color,
      getRadius: 50,
      radiusMinPixels: 5
    }),
    // rutas de vehículos
    new PathLayer({
      id: "trails",
      data: Object.values(vehicleData),
      getPath: d => d.trail,
      getColor: d => d.color,
      getWidth: 3
    })
  ];

  const overlay = {
    position: "absolute", top: 10, left: 10, zIndex: 100,
    backgroundColor: "rgba(255,255,255,0.8)", padding: 12,
    borderRadius: 4, fontSize: "0.9rem", maxWidth: 300
  };

  return (
    <>
      <div style={overlay}>
        <h2>Almacenes San José</h2>
        <ProductList products={products} />
        <TruckList trucks={trucks} />
      </div>

      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller
        layers={layers}
      >
        <Map reuseMaps mapLib={import("maplibre-gl")} mapStyle={MAP_STYLE} />
      </DeckGL>
    </>
  );
}

export default App;
