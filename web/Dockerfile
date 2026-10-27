FROM node:20-alpine

WORKDIR /app

COPY web/package.json web/package-lock.json* ./
RUN npm install

COPY web/ .

EXPOSE 5174

# API_URL is read by vite.config.ts at dev-server startup to set the proxy target.
# In Docker it must point to the api service, not localhost.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
