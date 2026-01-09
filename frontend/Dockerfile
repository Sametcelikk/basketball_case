FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY vite.config.ts tsconfig.json tsconfig.app.json tsconfig.node.json ./
COPY components.json* ./
COPY index.html ./

COPY src/ ./src/
COPY public/ ./public/

EXPOSE 5173

CMD ["npm", "run", "dev"]
