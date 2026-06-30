// Configuration for the chatbot frontend
const CONFIG = {
  // Backend API base URL (lokalni razvoj)
  API_BASE_URL: "http://localhost:8000",

  get API_URL() {
    return this.API_BASE_URL + "/api/chat";
  },
};
