const API_URL =
  import.meta.env.VITE_API_URL || "/api";

async function parseResponse(response) {
  if (!response.ok) {
    let message = "Request failed.";

    try {
      const error = await response.json();
      message = error.detail || message;
    } catch {
      message = await response.text();
    }

    throw new Error(message);
  }

  return response.json();
}

export async function predictPlate(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    `${API_URL}/predict`,
    {
      method: "POST",
      body: formData,
    }
  );

  return parseResponse(response);
}

export async function getReferenceStats() {
  const response = await fetch(
    `${API_URL}/reference-stats`
  );

  return parseResponse(response);
}