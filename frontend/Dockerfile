# ── Stage 1: Build the React app ───────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (layer cached unless package.json changes)
COPY package*.json ./
RUN npm ci --silent

# Copy source and build
COPY . .
RUN npm run build

# ── Stage 2: Serve with nginx ───────────────────────────────────────────────────
FROM nginx:1.25-alpine

# Replace default nginx config with ours
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built React files
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
