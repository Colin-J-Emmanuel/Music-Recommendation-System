require("dotenv").config(); //allows us to access our env variables within the following code 
const axios = require('axios'); //imports "axios" a library that simplifies HTTP request to external 
//simpler and more convienent (HTTP request GET, POST, PUT etc.) 

async function getAccessToken() //defining an asynchronus function the gets an AcessToken (hence the name)
//KEYWORD async: allows the function to use await inside its body making asynchronus calls
//An asynchronous (async) call allows our program to work on this function whilst other code runs, as javaScript is single-threaded
//(only executing one tak at a time in the main thread) 
{
    try//attempt to do this...
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
        }); //basically sends a POST request to Spotify Accounts service API to get an AccessToken
        console.log('Access Token:', response.data.access_token); //The access token being printed
    }
    catch(error)//...if not print this error
    {
        console.error('Error fetching acess token:', error); // the error that would be printed
    }
}

getAccessToken(); //calls our function to be run
