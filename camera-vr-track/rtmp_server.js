const NodeMediaServer = require('node-media-server');
const config = {
  rtmp: {
    port: 1935,
    chunk_size: 4096, // Reduziere die Chunk-Größe für schnellere Übertragung
    gop_cache: false, // Deaktiviere GOP-Cache, um Latenz zu verringern
    ping: 30,
    ping_timeout: 60,
    buffer: 0 // Setze den Puffer auf 0 für minimale Latenz
  }
};
new NodeMediaServer(config).run();