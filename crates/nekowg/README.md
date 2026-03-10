# Welcome to NekoWG!

NekoWG is a hybrid immediate and retained mode, GPU accelerated, UI framework
for Rust, designed to support a wide variety of applications.

## Getting Started

NekoWG is still in active development as we work on the Zed code editor, and is still pre-1.0. There will often be breaking changes between versions. You'll also need to use the latest version of stable Rust and be on macOS or Linux. Add the following to your `Cargo.toml`:

```toml
nekowg = { version = "*" }
```

 - [Ownership and data flow](_ownership_and_data_flow)

Everything in NekoWG starts with an `Application`. You can create one with `Application::new()`, and kick off your application by passing a callback to `Application::run()`. Inside this callback, you can create a new window with `App::open_window()`, and register your first root view. See [nekowg.rs](https://www.nekowg.rs/) for a complete example.

### Dependencies

NekoWG has various system dependencies that it needs in order to work.

#### macOS

On macOS, NekoWG uses Metal for rendering. In order to use Metal, you need to do the following:

- Install [Xcode](https://apps.apple.com/us/app/xcode/id497799835?mt=12) from the macOS App Store, or from the [Apple Developer](https://developer.apple.com/download/all/) website. Note this requires a developer account.

> Ensure you launch Xcode after installing, and install the macOS components, which is the default option.

- Install [Xcode command line tools](https://developer.apple.com/xcode/resources/)

  ```sh
  xcode-select --install
  ```

- Ensure that the Xcode command line tools are using your newly installed copy of Xcode:

  ```sh
  sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
  ```

## The Big Picture

NekoWG offers three different [registers](<https://en.wikipedia.org/wiki/Register_(sociolinguistics)>) depending on your needs:

- State management and communication with `Entity`'s. Whenever you need to store application state that communicates between different parts of your application, you'll want to use NekoWG's entities. Entities are owned by NekoWG and are only accessible through an owned smart pointer similar to an `Rc`. See the `app::context` module for more information.

- High level, declarative UI with views. All UI in NekoWG starts with a view. A view is simply an `Entity` that can be rendered, by implementing the `Render` trait. At the start of each frame, NekoWG will call this render method on the root view of a given window. Views build a tree of `elements`, lay them out and style them with a tailwind-style API, and then give them to NekoWG to turn into pixels. See the `div` element for an all purpose swiss-army knife of rendering.

- Low level, imperative UI with Elements. Elements are the building blocks of UI in NekoWG, and they provide a nice wrapper around an imperative API that provides as much flexibility and control as you need. Elements have total control over how they and their child elements are rendered and can be used for making efficient views into large lists, implement custom layouting for a code editor, and anything else you can think of. See the `element` module for more information.

Each of these registers has one or more corresponding contexts that can be accessed from all NekoWG services. This context is your main interface to NekoWG, and is used extensively throughout the framework.

## GPU Callbacks (Current-Target)

NekoWG exposes a low-level, scene-integrated GPU callback API for advanced rendering on the
current window target. This is useful for fullscreen effects, custom visualizations, and
GPU-accelerated drawing that does not fit into the retained rendering model.

High-level flow:

1. Implement `GpuNode` to prepare GPU resources and encode draw commands.
2. Register the node with `Window::insert_gpu_node`.
3. Schedule the node each frame via `Window::push_gpu_primitive`.

See `examples/gpu_primitive.rs` for a working end-to-end example.

## Other Resources

In addition to the systems above, NekoWG provides a range of smaller services that are useful for building complex applications:

- Actions are user-defined structs that are used for converting keystrokes into logical operations in your UI. Use this for implementing keyboard shortcuts, such as cmd-q. See the `action` module for more information.

- Platform services, such as `quit the app` or `open a URL` are available as methods on the `app::App`.

- An async executor that is integrated with the platform's event loop. See the `executor` module for more information.,

- The `[nekowg::test]` macro provides a convenient way to write tests for your NekoWG applications. Tests also have their own kind of context, a `TestAppContext` which provides ways of simulating common platform input. See `app::test_context` and `test` modules for more details.

Currently, the best way to learn about these APIs is to read the Zed source code or drop a question in the [Zed Discord](https://zed.dev/community-links). We're working on improving the documentation, creating more examples, and will be publishing more guides to NekoWG on our [blog](https://zed.dev/blog).

- [Low-level graphics API draft](docs/graphics_api_draft.md)
