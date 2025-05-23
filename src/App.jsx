import React, { useEffect, useState, useRef } from "react";
import { DeckGL } from "deck.gl";
import { Map } from "react-map-gl/maplibre";
import { IconLayer, ScatterplotLayer} from "@deck.gl/layers";


// Usa un mapa gratuito de MapLibre en lugar de requerir API key
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
  const [trafficLights, setTrafficLights] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState("Conectando...");
  const [errorMessage, setErrorMessage] = useState("");
  const trailsRef = useRef({});
  const wsRef = useRef(null);

  useEffect(() => {
    // Manejo mejorado de la conexión WebSocket
    const connectWebSocket = () => {
      try {
        console.log("Intentando conectar al WebSocket...");
        const ws = new WebSocket("ws://localhost:8765");
        wsRef.current = ws;

        ws.onopen = () => {
          console.log("Conexión WebSocket establecida");
          setConnectionStatus("Conectado");
          setErrorMessage("");
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log(`Recibidos datos para ${data.vehicles.length} vehículos`);

            const updated = {};
            data.vehicles.forEach((v) => {
              const prevTrail = trailsRef.current[v.id] || [];
              const newTrail = [...prevTrail.slice(-20), [v.lon, v.lat]];
              trailsRef.current[v.id] = newTrail;

              updated[v.id] = {
                id: v.id,
                position: [v.lon, v.lat],
                trail: newTrail,
                color: 0,
              };
            });

            setVehicleData(updated);
            setTrafficLights(data.traffic_lights || []);

          } catch (err) {
            console.error("Error procesando mensaje:", err);
            setErrorMessage(`Error procesando datos: ${err.message}`);
          }
        };

        ws.onclose = () => {
          console.log("Conexión WebSocket cerrada");
          setConnectionStatus("Desconectado");
          // Reintentar conexión después de 3 segundos
          setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
          console.error("Error en WebSocket:", error);
          setConnectionStatus("Error");
          setErrorMessage("Error en la conexión WebSocket");
        };
      } catch (err) {
        console.error("Error creando WebSocket:", err);
        setConnectionStatus("Error");
        setErrorMessage(`Error de conexión: ${err.message}`);
        // Reintentar conexión después de 3 segundos
        setTimeout(connectWebSocket, 3000);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);


  const layers = [
    new IconLayer({
      id: "vehicle-icons",
      data: Object.values(vehicleData),
      getIcon: (d) => ({
        url: "/icons/car.png",
        width: 128,
        height: 128,
        anchorY: 128,
      }),
      getPosition: (d) => d.position,
      getSize: 2, // escala del ícono
      sizeScale: 10,
      getAngle: 0,
      getColor: 0,
      pickable: true,
    }),

    new ScatterplotLayer({
      id: "traffic-lights",
      data: trafficLights,
      getPosition: (d) => [d.lon, d.lat],
      getFillColor: (d) => d.state === "red" ? [255, 0, 0] : [0, 255, 0],
      getRadius: 10,
      radiusMinPixels: 2,
    }),


  ];

  return (
    <>
      <div
        style={{
          position: "absolute",
          top: "10px",
          left: "10px",
          zIndex: 100,
          padding: "10px",
          backgroundColor: "rgba(255,255,255,0.7)",
          borderRadius: "4px",
          maxWidth: "300px",
        }}
      >
        <div>
          <strong>Estado:</strong> {connectionStatus}
        </div>
        {errorMessage && (
          <div style={{ color: "red" }}>{errorMessage}</div>
        )}
        <div>
          <strong>Vehículos:</strong> {Object.keys(vehicleData).length}
        </div>
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
