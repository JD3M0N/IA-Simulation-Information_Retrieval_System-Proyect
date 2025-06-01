import React, { useEffect, useState, useRef } from "react";
import { DeckGL } from "deck.gl";
import { Map } from "react-map-gl/maplibre";
import { IconLayer, ScatterplotLayer, PathLayer } from "@deck.gl/layers";
import OptimizationForm from "./components/OptimizationForm";
import RouteInfo from "./components/RouteInfo";
import StatusPanel from "./components/StatusPanel";
import StatsButton from "./components/StatsButton";
import StatsModal from "./components/StatsModal";
import { calculateRouteDistance } from "./utils/distance";
import { getRouteColor } from "./utils/colors";

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
  const [optimizedRoutes, setOptimizedRoutes] = useState([]);
  const [showOptimizationForm, setShowOptimizationForm] = useState(false);
  const [optimizationParams, setOptimizationParams] = useState({
    start_point: "",
    target_points: [],
    num_trucks: 3,
    truck_capacities: [10, 10, 10],
    target_demands: {}
  });
  const [selectedRouteId, setSelectedRouteId] = useState(null);
  const [showDetailedStats, setShowDetailedStats] = useState(false);
  
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
            
            // Si es una actualización de posiciones
            if (data.vehicles) {
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
            }
            
            // Si es una respuesta de optimización
            if (data.type === 'optimization_result') {
              console.log("Recibidos resultados de optimización:", data);
              
              // Transformar las rutas al formato esperado por PathLayer
              // y calcular la distancia una sola vez
              const routes = data.routes.map((route, idx) => {
                const path = route.map(point => [point.lon, point.lat]);
                return {
                  id: `route-${idx}`,
                  path: path,
                  color: getRouteColor(idx),
                  distance: calculateRouteDistance(path), // Calculamos la distancia una sola vez
                  vehicleId: `Camión ${idx + 1}`
                };
              });
              
              setOptimizedRoutes(routes);
            }
            
            // Si hay un error de optimización
            if (data.type === 'optimization_error') {
              setErrorMessage(data.message);
            }
            
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

  // Función para enviar solicitud de optimización
  const requestOptimization = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'optimization_request',
        ...optimizationParams
      }));
    } else {
      setErrorMessage("No hay conexión con el servidor");
    }
  };
  

  // Función para manejar la selección de rutas
  const handleRouteSelect = (routeId) => {
    setSelectedRouteId(routeId === selectedRouteId ? null : routeId);
  };

  // Preparar rutas con propiedad de selección
  const routesWithSelection = optimizedRoutes.map(route => ({
    ...route,
    selected: route.id === selectedRouteId
  }));

  const layers = [
    // Capa para rutas optimizadas
    new PathLayer({
      id: 'optimized-routes',
      data: optimizedRoutes,
      pickable: true,
      widthScale: 1,
      widthMinPixels: 2,
      getPath: d => d.path,
      getColor: d => d.id === selectedRouteId ? [255, 255, 255] : d.color,
      getWidth: d => d.id === selectedRouteId ? 8 : 5,
      onHover: (info) => {
        // Actualiza el tooltip si se necesita
      },
      // Información que se mostrará al pasar el cursor
      getTooltip: (obj) => {
        if (obj.object) {
          return {
            html: `
              <div>
                <b>${obj.object.vehicleId}</b><br/>
                Distancia: ${obj.object.distance.toFixed(2)} km<br/>
                Puntos: ${obj.object.path.length}
              </div>
            `
          };
        }
        return null;
      }
    }),
    
    // Capas existentes
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
  
  // Agregamos un panel para mostrar la información de las rutas
  return (
    <>
      {/* Panel de estado extraído como componente */}
      <StatusPanel 
        connectionStatus={connectionStatus}
        errorMessage={errorMessage}
        vehicleCount={Object.keys(vehicleData).length}
        showOptimizationForm={showOptimizationForm}
        setShowOptimizationForm={setShowOptimizationForm}
      />
      
      {/* Formulario de optimización */}
      {showOptimizationForm && 
        <OptimizationForm 
          optimizationParams={optimizationParams}
          setOptimizationParams={setOptimizationParams}
          requestOptimization={requestOptimization}
          setOptimizedRoutes={setOptimizedRoutes}
        />
      }
      
      {/* Panel mejorado para mostrar información de rutas */}
      {optimizedRoutes.length > 0 && (
        <>
          <RouteInfo 
            routes={routesWithSelection} 
            onSelectRoute={handleRouteSelect} 
          />
          
          {/* Botón para mostrar/ocultar estadísticas detalladas */}
          <StatsButton 
            showDetailedStats={showDetailedStats}
            setShowDetailedStats={setShowDetailedStats}
          />
          
          {/* Panel de estadísticas detalladas */}
          <StatsModal 
            showDetailedStats={showDetailedStats}
            setShowDetailedStats={setShowDetailedStats}
            optimizedRoutes={optimizedRoutes}
          />
        </>
      )}
      
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
