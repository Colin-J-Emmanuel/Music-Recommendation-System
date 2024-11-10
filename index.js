require("dotenv").config(); //allows us to access our env variables within the following code 
const axios = require('axios'); 

async function getAccessToken()
{
    try
    {
        const response = await axios.post('https://accounts.spotify.com/api/token', null,
        {
            params: 
            {
                grant_type: 'client_credentials'
            }, 
            headers: 
            {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': `Basic ${Buffer.from(`${process.env.SPOTIFY_CLIENT_ID}:${process.env.SPOTIFY_CLIENT_SECRET}`).toString('base64')}`
            }
        });
        console.log('Access Token:', response.data.access_token);
    }
    catch(error)
    {
        console.error('Error fetching acess token:', error); 
    }
}

getAccessToken();
